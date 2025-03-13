# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from inspect import signature
from typing import Union, List
import mmcv
import torch
import torch.nn as nn
from mmcv.image import tensor2imgs
import clip

from mmdet.core import bbox_mapping
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.datasets.lvis import LVISV1Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

# def load_clip_to_cpu(cfg):
#     backbone_name = cfg.MODEL.BACKBONE.NAME
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")

#     model = clip.build_model(state_dict or model.state_dict())

#     return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, ctx_len, category_descriptions, clip_model):
        super().__init__()
        classnames = [category_description.split("_which_is_")[0] for category_description in category_descriptions]
        classdefs = [category_description.split("_which_is_")[1] for category_description in category_descriptions]

        n_cls = len(classnames)
        n_ctx1 = ctx_len
        n_ctx2 = ctx_len
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        print("Initializing a generic context")
        ctx_vectors1 = torch.empty(n_ctx1, ctx_dim, dtype=dtype)
        ctx_vectors2 = torch.empty(n_ctx2, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors1, std=0.02)
        nn.init.normal_(ctx_vectors2, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx1)
        prompt_suffix = " ".join(["X"] * n_ctx2)

        print(f'Initial context1: "{prompt_prefix}"')
        print(f'Initial context2: "{prompt_suffix}"')
        print(f"Number of context words (tokens1): {n_ctx1}")
        print(f"Number of context words (tokens2): {n_ctx2}")

        self.ctx1 = nn.Parameter(ctx_vectors1)  # to be optimized
        self.ctx2 = nn.Parameter(ctx_vectors2)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        classdefs = [name.replace("_", " ") for name in classdefs]
        def_lens = [len(_tokenizer.encode(name)) for name in classdefs]

        prompts = [prompt_prefix + " " + name + " " + prompt_suffix + " " + classdef + "." for name, classdef in
                   zip(classnames, classdefs)]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("embedding", embedding)  # SOS

        self.n_cls = n_cls
        self.n_ctx1 = n_ctx1
        self.n_ctx2 = n_ctx2
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.def_lens = def_lens

    def forward(self):
        ctx1 = self.ctx1
        ctx2 = self.ctx2
        if ctx1.dim() == 2:
            ctx1 = ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx2 = ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)

        embedding = self.embedding

        n_ctx1 = self.n_ctx1
        n_ctx2 = self.n_ctx2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            def_len = self.def_lens[i]
            prefix_i = embedding[i: i + 1, :1, :]
            ctx1_i = ctx1[i:i + 1]
            name_index = 1 + n_ctx1
            name_i = embedding[i: i + 1, name_index:name_index + name_len, :]
            ctx2_i = ctx2[i:i + 1]
            def_index = 1 + n_ctx1 + name_len + n_ctx2
            def_i = embedding[i: i + 1, def_index:def_index + def_len, :]
            suf_index = 1 + n_ctx1 + name_len + n_ctx2 + def_len
            suffix_i = embedding[i: i + 1, suf_index:, :]
            prompt = torch.cat(
                [
                    prefix_i,
                    ctx1_i,
                    name_i,
                    ctx2_i,
                    def_i,
                    suffix_i,
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, ctx_len, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(ctx_len, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits




@DETECTORS.register_module()
class SourcingPromptLearner(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 clip_checkponint,
                 ctx_len,
                 class_file,
                 train_cfg,
                 test_cfg,
                 init_cfg=None):
        super(SourcingPromptLearner, self).__init__(init_cfg)

        classnames = []
        with open(class_file, 'r') as f:
            for eachline in f:
                eachline = eachline.strip('\n')#去掉末尾的换行符
                eachline = eachline.strip(' ')#去掉首尾多于的空格
                eachline = eachline.split(' ')
                if len(eachline):
                    classnames.append(eachline[1])

        model, preprocess = clip.load(clip_checkponint, device=super.device)

        print("Building custom CLIP")
        self.model = CustomCLIP(ctx_len, classnames, model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Extract features.

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        pass

    def forward_dummy(self, img):
        """Dummy forward function."""
        
        return self.model(img)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        logits = self.model(img)

        losses = self.rpn_head.forward_train(x, img_metas, gt_bboxes, None,
                                             gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        x = self.extract_feat(img)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
        if torch.onnx.is_in_onnx_export():
            return proposal_list

        return [proposal.cpu().numpy() for proposal in proposal_list]

    def aug_test(self, imgs, img_metas, rescale=False):
        pass

    
