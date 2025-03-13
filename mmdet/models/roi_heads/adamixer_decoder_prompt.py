import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh, multiclass_nms, bbox_overlaps, reduce_mean
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS, build_loss
from .cascade_roi_head import CascadeRoIHead


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            # in_proj_weight=torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]),
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)



class Queues(nn.Module):
    def __init__(self, names, lengths, emb_dim=512):
        super(Queues, self).__init__()
        self.names = names
        self.lengths = lengths
        self.emb_dim = emb_dim
        self._init_queues()

    def _init_queues(self):
        attr_names = self.names
        queue_lengths = self.lengths
        for n in attr_names:
            self.register_buffer(n, torch.ones(0, self.emb_dim), persistent=False)
        self.queue_lengths = {n: queue_lengths[i] for i, n in enumerate(attr_names)}

    @torch.no_grad()
    def dequeue_and_enqueue(self, queue_update):
        for k, feat in queue_update.items():
            if len(feat) == 0:
                continue
            queue_length = self.queue_lengths[k]
            in_length = feat.shape[0]
            queue_value = getattr(self, k)
            current_length = queue_value.shape[0]
            kept_length = min(queue_length - in_length, current_length)

            queue_value.data = torch.cat([feat, queue_value[:kept_length]])

    @torch.no_grad()
    def get_queue(self, key):
        value = getattr(self, key)
        return value


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activations = nn.ModuleList(nn.GELU() for _ in range(num_layers - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class AdaMixerDecoderPrompt(CascadeRoIHead):
    _DEBUG = -1

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 clip_model_path=None,
                 cls_tau=100,
                 skd_tau=20,
                 rkd_tau=5,
                 alpha=0.35,
                 beta=0.65,
                 text_dim=512,
                 num_additional_padding_prompts=32,
                 similar_weight=1.0,
                 use_pseudo_box=True,
                 split_visual_text=False,
                 loss_visual_skd=dict(type='CrossEntropyLoss', loss_weight=2.0),
                 loss_visual_rkd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=2.0),
                 novel_obj_queue_dict=None,
                 pre_extracted_clip_text_feat=None,
                 use_text_space_rkd_loss=False,
                 use_image_level_distill=False,
                 max_pseudo_box_num=5,
                 image_distill='v1',
                 bbox_head=None,
                 mask_head=None,
                 mask_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        clip_model_size = clip_model_path.split('/')[-1]
        if clip_model_size == 'CLIP_RN50.pt':
            self.pooling_size = 7
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=self.pooling_size, sampling_ratio=2),
                out_channels=2048,
                featmap_strides=[32])
        elif clip_model_size == 'CLIP_RN50x4.pt':
            self.pooling_size = 9
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=self.pooling_size, sampling_ratio=2),
                out_channels=2560,
                featmap_strides=[32])
        elif clip_model_size == 'CLIP_RN50x16.pt':
            self.pooling_size = 12
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=self.pooling_size, sampling_ratio=2),
                out_channels=3072,
                featmap_strides=[32])
        elif clip_model_size == 'CLIP_ViT-B-32.pt':
            self.pooling_size = 7
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[32])
        else:
            raise NotImplementedError
            
        super(AdaMixerDecoderPrompt, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            mask_roi_extractor=mask_roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.max_pseudo_box_num = max_pseudo_box_num
        self.tau = cls_tau
        self.skd_tau = skd_tau
        self.rkd_tau = rkd_tau
        self.use_mask_head = True if mask_head is not None else False
        if novel_obj_queue_dict is not None:
            self.queue = Queues(**novel_obj_queue_dict)

        self.use_pseudo_box = use_pseudo_box
        self.bg_embedding = nn.Embedding(1, text_dim)

        self.split_visual_text = split_visual_text
        if self.split_visual_text:
            self.visual2text = MLP(text_dim, text_dim*2, text_dim, 3)
            self.fc1 = nn.Linear(text_dim*2, text_dim*2)
            self.fc2 = nn.Linear(text_dim*2, text_dim)
            self.relu = nn.ReLU()

        self.num_additional_padding_prompts = num_additional_padding_prompts
        self.alpha = alpha
        self.beta = beta
        self.image_id = 0
        self.image_id2 = 0
        if use_pseudo_box:
            self.loss_visual_skd = build_loss(loss_visual_skd)
            self.loss_visual_rkd = build_loss(loss_visual_rkd)
        self.use_pre_extracted_clip_text_feat = pre_extracted_clip_text_feat is not None
        self.pre_extracted_clip_text_feat_path = pre_extracted_clip_text_feat
        self.use_text_space_rkd_loss = use_text_space_rkd_loss
        if self.use_text_space_rkd_loss:
            self.class_centroid = None
            self.category_embeddings = None
            self.loss_text_skd = build_loss(loss_visual_skd)
            self.loss_text_rkd = build_loss(loss_visual_rkd)
        self.use_image_level_distill = use_image_level_distill
        self.image_distill = image_distill
        if self.use_image_level_distill:
            if self.image_distill == 'v1':
                self.linear_transform = nn.Linear(content_dim, text_dim)
                self.layernorm = nn.LayerNorm(content_dim)
            elif self.image_distill == 'v2':
                self.linear_transform = nn.Linear(content_dim * 49, text_dim)
                self.layernorm = nn.LayerNorm(content_dim * 49)
            else:
                raise NotImplementedError
            self.loss_img_distill = build_loss(dict(type='CrossEntropyLoss', loss_weight=0.2))

    @torch.no_grad()
    def init_weights(self):
        super(AdaMixerDecoderPrompt, self).init_weights()
        if hasattr(self, 'bbox_head'):
            for i in range(len(self.bbox_head)):
                self.bbox_head[i].init_weights()
        if hasattr(self, 'mask_head'):
            for i in range(len(self.mask_head)):
                self.mask_head[i].init_weights()
        if self.split_visual_text:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def draw_boxes(self, stage, image, boxes, gt_bbox):
        image = image.to(torch.uint8).cpu()
        H, W = image.shape[-2:]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=H)
        img = image.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save('%d_%d_ori_img.jpeg'%(self.image_id, stage))
        img = torchvision.utils.draw_bounding_boxes(image, boxes, colors='blue', width=6)
        # img = torchvision.utils.draw_bounding_boxes(img, all_bbox)
        img = torchvision.utils.draw_bounding_boxes(img, gt_bbox, colors='red', width=6)
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img.save('%d_%d_all_box.jpeg'%(self.image_id, stage))

        self.image_id += 1

    def xyxy2xyzr(self, bbox):
        xy = 0.5 * (bbox[..., 0:2] + bbox[..., 2:4])
        wh = bbox[..., 2:4] - bbox[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()
        xyzr = torch.cat([xy, z, r], dim=-1)
        return xyzr
    
    def _mask_forward_train(self, stage, x, attn_feats, sampling_results,
                            gt_masks, rcnn_train_cfg):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        attn_feats = torch.cat([
            feats[res.pos_inds]
            for (feats, res) in zip(attn_feats, sampling_results)
        ])
        mask_results = self._mask_forward(stage, x, pos_rois, attn_feats)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results]) * 0

        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)
        mask_results.update(loss_mask)
        return mask_results

    def _mask_forward(self, stage, x, rois, attn_feats):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results
    

    def _bbox_forward(self, stage, img_feat, query_xyzr, query_content, img_metas):

        bbox_head = self.bbox_head[stage]
        attn_feats, cls_score_feature, delta_xyzr, query_content, objness = bbox_head(img_feat, query_xyzr, query_content,
                                                         featmap_strides=self.featmap_strides, img_metas=img_metas)

        query_xyzr, decoded_bboxes = self.bbox_head[stage].refine_xyzr(
            query_xyzr,
            delta_xyzr)

        bboxes_list = [bboxes for bboxes in decoded_bboxes]

        bbox_results = dict(
            attn_feats=attn_feats,
            cls_score_feature=cls_score_feature,
            query_xyzr=query_xyzr,
            decode_bbox_pred=decoded_bboxes,
            query_content=query_content,
            # detach_cls_score_list=[
            #     cls_score[i].detach() for i in range(num_imgs)
            # ],
            detach_bboxes_list=[item.detach() for item in bboxes_list],
            bboxes_list=bboxes_list,
            objness=objness,
        )
        
        return bbox_results


    def forward_train(self,
                      x,
                      backbone_features, 
                      img_no_normalize,
                      proposals,
                      query_content,
                      all_pseudo_boxes,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      remapping_gt_labels,
                      category_embeddings,
                      idxs,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        num_imgs = len(img_metas)
        num_pos = [len(gt_label) for gt_label in gt_labels]
        num_pos = torch.tensor(num_pos, dtype=torch.float, device=query_content.device).sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1)
        num_base_classes = category_embeddings.shape[1]

        query_xyzr = self.xyxy2xyzr(proposals).detach()

        num_queries = query_xyzr.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_queries, 1)
        all_stage_bbox_results = []
        all_stage_loss = {}
        all_stage_assign_results = []

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyzr, query_content,
                                              img_metas)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            bboxes_list = bbox_results['detach_bboxes_list']
            query_xyzr = bbox_results['query_xyzr'].detach()
            query_content = bbox_results['query_content']
            objness = bbox_results['objness']

            sampling_results = []
            assign_results = []
            with torch.no_grad():
                for i in range(num_imgs):
                    normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_list[i] / imgs_whwh[i])
                    # class-agnostic assign
                    assign_result = self.bbox_assigner[stage].assign(
                        normalize_bbox_ccwh, objness[i].clone().detach(), gt_bboxes[i],
                        remapping_gt_labels[i] * 0, img_metas[i])
                    
                    # changing assign_result back to class-specific
                    matched_row_inds = assign_result.gt_inds > 0
                    matched_col_inds = assign_result.gt_inds[matched_row_inds] - 1
                    assign_result.labels[matched_row_inds] = remapping_gt_labels[i][matched_col_inds]

                    sampling_result = self.bbox_sampler[stage].sample(
                        assign_result, bboxes_list[i], gt_bboxes[i])

                    sampling_results.append(sampling_result)
                    assign_results.append(assign_result)

            all_stage_assign_results.append(assign_results)
            if self.use_mask_head:
                mask_results = self._mask_forward_train(stage, x[:self.mask_roi_extractor[stage].num_inputs], bbox_results['attn_feats'], sampling_results,
                    gt_masks, self.train_cfg[stage])
                all_stage_loss[f'stage{stage}_loss_mask'] = mask_results['loss_mask'] * self.stage_loss_weights[stage]

        # get pseudo box embedding
        gt_bboxes_with_pseudo_box = []
        gt_labels_with_pseudo_box = []
        all_weighting_score = []
        gt_bboxes_clip_image_feature = []
        all_pseudo_boxes_clip_image_feature = []
        all_pseudo_box_clip_embedding = []
        clip_full_image_embedding = []

        with torch.no_grad():
            for i in range(num_imgs):
                filename = img_metas[i]['ori_filename']
                filename = filename.split('/')[-1]
                filename = filename[:-4]
                pre_extracted_embedding = torch.load(self.pre_extracted_clip_text_feat_path + filename + '.pth', 'cpu')
                clip_full_image_embedding.append(pre_extracted_embedding['clip_full_image_embedding'])
                if not self.use_pseudo_box:
                    gt_bboxes_with_pseudo_box = gt_bboxes
                    gt_labels_with_pseudo_box = remapping_gt_labels
                    all_weighting_score.append(torch.zeros((0,), device=category_embeddings.device))
                else:
                    gt_clip_image_feature = pre_extracted_embedding['gt_clip_image_embedding'].to(category_embeddings.device).to(category_embeddings.dtype)
                    gt_clip_image_feature = F.normalize(gt_clip_image_feature, dim=-1)
                    gt_bboxes_clip_image_feature.append(gt_clip_image_feature)

                    size_mask = ((all_pseudo_boxes[i][:, 2] - all_pseudo_boxes[i][:, 0]) > 32) & ((all_pseudo_boxes[i][:, 3] - all_pseudo_boxes[i][:, 1]) > 32)
                    all_pseudo_boxes[i] = all_pseudo_boxes[i][size_mask][:self.max_pseudo_box_num]

                    pseudo_box_clip_image_feature = pre_extracted_embedding['proposal_clip_image_embedding'].to(category_embeddings.device).to(category_embeddings.dtype)[size_mask][:self.max_pseudo_box_num]
                    pseudo_box_clip_image_feature = F.normalize(pseudo_box_clip_image_feature, dim=-1)
                    
                    if self.use_pre_extracted_clip_text_feat:
                        pseudo_box_clip_text_feature = pre_extracted_embedding['proposal_clip_text_embedding'].to(category_embeddings.device).to(category_embeddings.dtype)[size_mask][:self.max_pseudo_box_num]
                        pseudo_box_clip_text_feature = F.normalize(pseudo_box_clip_text_feature, dim=-1)
                    else:
                        pseudo_box_clip_text_feature = pseudo_box_clip_image_feature
                        
                    
                    # weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.2) / 0.03) # R50
                    # weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.25) / 0.026) # ViT-B-32
                    weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.31) / 0.026) * 1.3 # ViT-B-32 detpro
                    #weighting_mask = weighting_score > 0.3
                    #all_pseudo_boxes_clip_image_feature.append(pseudo_box_clip_image_feature[weighting_mask])
                    #all_pseudo_box_clip_embedding.append(pseudo_box_clip_text_feature[weighting_mask])
                    #all_weighting_score.append(weighting_score[weighting_mask])
                    #all_pseudo_boxes[i] = all_pseudo_boxes[i][weighting_mask]
                    all_pseudo_boxes_clip_image_feature.append(pseudo_box_clip_image_feature)
                    all_pseudo_box_clip_embedding.append(pseudo_box_clip_text_feature)
                    all_weighting_score.append(weighting_score)

                    gt_bboxes_with_pseudo_box.append(torch.cat([gt_bboxes[i], all_pseudo_boxes[i]]))
                    pseudo_box_label = torch.arange(0, len(all_pseudo_boxes[i]), device=all_pseudo_boxes[i].device, dtype=gt_labels[0].dtype) + num_base_classes
                    gt_labels_with_pseudo_box.append(torch.cat([remapping_gt_labels[i], pseudo_box_label]))
            if len(clip_full_image_embedding) > 0:
                clip_full_image_embedding = torch.cat(clip_full_image_embedding, dim=0).to(category_embeddings.device).to(category_embeddings.dtype)
                clip_full_image_embedding = F.normalize(clip_full_image_embedding, dim=-1)
        
        # image_level_distill
        if self.use_pseudo_box and self.use_image_level_distill:
            img_distill_loss = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
            image_box = [torch.tensor([[0, 0, img_meta['img_shape'][1], img_meta['img_shape'][0]]], dtype=gt_bboxes[0].dtype, device=gt_bboxes[0].device) for img_meta in img_metas]
            fused_x = x[3]
            for i in range(3):
                fused_x = fused_x + F.interpolate(x[i], fused_x.shape[-2:], mode='bilinear')
            image_box = bbox2roi(image_box)
            if self.image_distill == 'v1':
                backbone_image_features = self.bbox_roi_extractor[0]([fused_x], image_box).reshape(num_imgs, self.content_dim, self.pooling_size*self.pooling_size)
                backbone_image_features = torch.mean(backbone_image_features, dim=-1)
            elif self.image_distill == 'v2':
                backbone_image_features = self.bbox_roi_extractor[0]([fused_x], image_box).reshape(num_imgs, self.content_dim, self.pooling_size*self.pooling_size).flatten(1)
            else:
                raise NotImplementedError
            backbone_image_features = self.layernorm(backbone_image_features)
            backbone_image_features = self.linear_transform(backbone_image_features)
            backbone_image_features = F.normalize(backbone_image_features, dim=-1)

            saved_backbone_image_query = self.queue.get_queue('backbone_image_query')
            saved_clip_image_query = self.queue.get_queue('clip_image_query')
            skd_logits_1 = backbone_image_features @ torch.cat([clip_full_image_embedding, saved_clip_image_query]).t() * self.skd_tau
            skd_logits_2 = clip_full_image_embedding @ torch.cat([backbone_image_features, saved_backbone_image_query]).t() * self.skd_tau
            img_distill_loss = img_distill_loss + 0.5 * self.loss_img_distill(
                skd_logits_1,
                torch.arange(0, len(clip_full_image_embedding), device=clip_full_image_embedding.device, dtype=torch.long),)
            img_distill_loss = img_distill_loss + 0.5 * self.loss_img_distill(
                skd_logits_2,
                torch.arange(0, len(clip_full_image_embedding), device=clip_full_image_embedding.device, dtype=torch.long),)
            self.queue.dequeue_and_enqueue({'backbone_image_query': backbone_image_features.detach()})
            self.queue.dequeue_and_enqueue({'clip_image_query': clip_full_image_embedding.detach()})
            all_stage_loss['img_distill_loss'] = img_distill_loss

 
        # expand class prompt
        expanded_class_prompt = []
        if self.use_pseudo_box:
            for i in range(num_imgs):
                pseudo_box_clip_embedding = all_pseudo_box_clip_embedding[i:] + all_pseudo_box_clip_embedding[:i]
                pseudo_box_clip_embedding = torch.cat(pseudo_box_clip_embedding)
                expanded_class_prompt.append(torch.cat([category_embeddings[i], pseudo_box_clip_embedding, self.bg_embedding(torch.zeros((1,), device=query_content.device, dtype=torch.long))]))
            num_classes = len(pseudo_box_clip_embedding) + num_base_classes
            prompt_len = [len(prompt) for prompt in expanded_class_prompt]
            prompt_len = min(prompt_len)
            expanded_class_prompt = [prompt[:prompt_len] for prompt in expanded_class_prompt]
        else:
            for i in range(num_imgs):
                expanded_class_prompt.append(torch.cat([category_embeddings[i], self.bg_embedding(torch.zeros((1,), device=query_content.device, dtype=torch.long)),]))
            num_classes = num_base_classes

        class_prompt = torch.stack(expanded_class_prompt)
        all_stage_loss['num_pseudo_box'] = torch.tensor(num_classes - num_base_classes, dtype=torch.float, device=x[0].device) / num_imgs
        # all_stage_loss['num_text_pseudo_box'] = torch.sum(torch.cat(all_weighting_score) > 0.4).detach() / num_imgs

        # calculating losses
        for stage in range(self.num_stages):
            if self.use_text_space_rkd_loss:
                total_unique_label = torch.unique(torch.cat(gt_labels))
                if len(total_unique_label) >= 2:
                    base_foreground_query_mask = torch.cat([assign_result.gt_inds > 0 for assign_result in all_stage_assign_results[stage]])
                    gt_num = [len(gt_label) for gt_label in gt_labels]
                    gt_num = [0] + gt_num[:-1]
                    gt_num = torch.cumsum(torch.tensor(gt_num, device=base_foreground_query_mask.device), dim=0)
                    base_foreground_query_matched_inds = torch.cat([assign_result.gt_inds + gt_num[ii] for ii, assign_result in enumerate(all_stage_assign_results[stage])])[base_foreground_query_mask] - 1
                    total_gt_labels = torch.cat(gt_labels, dim=0)
                    total_gt_boxes = torch.cat(gt_bboxes, dim=0)

            # image space distillaton loss
            if self.use_pseudo_box:
                bboxes_list = all_stage_bbox_results[stage]['detach_bboxes_list']
                objness = all_stage_bbox_results[stage]['objness'].clone().detach()
                all_query_embedding = []
                all_clip_embedding = []
                for i in range(num_imgs):
                    assign_result = all_stage_assign_results[stage][i]
                    # two-stage assign
                    if len(all_pseudo_boxes[i]) > 0:
                        with torch.no_grad():
                            pseudo_box_label = torch.arange(0, len(all_pseudo_boxes[i]), device=all_pseudo_boxes[i].device, dtype=gt_labels[0].dtype) + num_base_classes
                            query_inds = torch.arange(0, num_queries, dtype=torch.long, device=all_pseudo_boxes[i].device)
                            query_inds = query_inds[assign_result.gt_inds == 0]

                            normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_list[i] / imgs_whwh[i])[query_inds]
                            assign_result_second = self.bbox_assigner[stage].assign(
                                normalize_bbox_ccwh, objness[i][query_inds], all_pseudo_boxes[i],
                                pseudo_box_label * 0, img_metas[i])
                            
                            # changing assign_result back to class-specific
                            matched_row_inds = assign_result_second.gt_inds > 0
                            matched_col_inds = assign_result_second.gt_inds[matched_row_inds] - 1
                            assign_result_second.labels[matched_row_inds] = pseudo_box_label[matched_col_inds]
                            assign_result_second.gt_inds[matched_row_inds] += len(gt_bboxes[i])

                            # merging assignment result
                            assign_result.labels[query_inds] = assign_result_second.labels
                            assign_result.gt_inds[query_inds] = assign_result_second.gt_inds
                            all_stage_assign_results[stage][i] = assign_result

                    if torch.sum(assign_result.gt_inds > 0) > 0:
                        query_embedding = all_stage_bbox_results[stage]['cls_score_feature'][i, assign_result.gt_inds > 0]
                        query_embedding = F.normalize(query_embedding, dim=-1)
                        
                        clip_image_feature_i = torch.cat([gt_bboxes_clip_image_feature[i], all_pseudo_boxes_clip_image_feature[i]])
                        inds = assign_result.gt_inds.clone()
                        inds[inds > 0] -= 1
                        clip_image_feature_i = clip_image_feature_i[inds]
                        clip_embedding = clip_image_feature_i[assign_result.gt_inds > 0]

                        matched_gt_box = gt_bboxes_with_pseudo_box[i][inds][assign_result.gt_inds > 0]
                        matched_pred_box = bboxes_list[i][assign_result.gt_inds > 0]
                        ious_mask = bbox_overlaps(matched_pred_box, matched_gt_box, is_aligned=True) > 0.5
                        all_query_embedding.append(query_embedding[ious_mask])
                        all_clip_embedding.append(clip_embedding[ious_mask])
                
                if len(all_clip_embedding) > 0:
                    all_clip_embedding = torch.cat(all_clip_embedding)
                    all_query_embedding = torch.cat(all_query_embedding)

                loss_rkd = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                loss_skd = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                if len(all_clip_embedding) >= 2:
                    saved_obj_query = self.queue.get_queue('obj_query_%d'%(stage))
                    saved_clip_embedding = self.queue.get_queue('clip_query_%d'%(stage))
                    # skd_logits_1 = all_query_embedding @ all_clip_embedding.t() * self.skd_tau
                    skd_logits_1 = all_query_embedding @ torch.cat([all_clip_embedding, saved_clip_embedding]).t() * self.skd_tau
                    skd_logits_2 = all_clip_embedding @ torch.cat([all_query_embedding, saved_obj_query]).t() * self.skd_tau
                    rkd_logits_clip = all_clip_embedding @ all_clip_embedding.t() * self.rkd_tau
                    rkd_logits_pred = all_query_embedding @ all_query_embedding.t() * self.rkd_tau
                    loss_skd = loss_skd + 0.5 * self.loss_visual_skd(
                        skd_logits_1,
                        torch.arange(0, len(all_clip_embedding), device=all_clip_embedding.device, dtype=torch.long),)
                    loss_skd = loss_skd + 0.5 * self.loss_visual_skd(
                        skd_logits_2,
                        torch.arange(0, len(all_clip_embedding), device=all_clip_embedding.device, dtype=torch.long),)
                    loss_rkd = loss_rkd + self.loss_visual_rkd(
                        rkd_logits_pred,
                        rkd_logits_clip,)
                    # loss_rkd = loss_rkd + self.irm_loss(all_query_embedding, all_clip_embedding) * self.loss_visual_rkd.loss_weight
                    self.queue.dequeue_and_enqueue({'obj_query_%d'%(stage): all_query_embedding.detach()})
                    self.queue.dequeue_and_enqueue({'clip_query_%d'%(stage): all_clip_embedding.detach()})

                all_stage_loss[f'stage{stage}_loss_skd'] = loss_skd * self.stage_loss_weights[stage]
                all_stage_loss[f'stage{stage}_loss_rkd'] = loss_rkd * self.stage_loss_weights[stage]

            
            bbox_results = all_stage_bbox_results[stage]
            cls_score_feature = bbox_results['cls_score_feature']
            if self.split_visual_text:
                t = self.fc1(self.visual2text.layers[-1].weight.clone().detach())
                t_act = self.relu(t)
                transfer_weights = self.fc2(t_act)
                cls_score_feature_rkd = cls_score_feature + F.linear(cls_score_feature, weight=transfer_weights)
                cls_score_feature = cls_score_feature + self.visual2text(cls_score_feature)
            
            # text space distillaton loss
            if self.use_text_space_rkd_loss:
                text_space_rkd_loss = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                text_space_skd_loss = torch.zeros(1, dtype=torch.float, device=query_xyzr.device)
                if len(total_unique_label) >= 2:
                    query_text_embedding = []
                    # base class query
                    foreground_query = cls_score_feature_rkd.flatten(0, 1)[base_foreground_query_mask]
                    with torch.no_grad():
                        all_pred_bbox = torch.cat(bbox_results['detach_bboxes_list'])[base_foreground_query_mask]
                        ious = bbox_overlaps(all_pred_bbox, total_gt_boxes[base_foreground_query_matched_inds], is_aligned=True).detach()
                    for label_id in total_unique_label:
                        class_mask = total_gt_labels[base_foreground_query_matched_inds] == label_id
                        matched_query = foreground_query[class_mask]
                        class_iou =  torch.softmax(ious[class_mask], dim=0)
                        fused_class_query = torch.einsum('nh,n->h', matched_query, class_iou)
                        query_text_embedding.append(fused_class_query)
                    query_text_embedding = torch.stack(query_text_embedding, dim=0)
                    normalize_query_text_embedding = F.normalize(query_text_embedding, dim=-1)

                    # novel class query
                    novel_query_embedding = []
                    novel_text_embedding = []
                    novel_query_weight = []
                    for i in range(num_imgs):
                        assign_result = all_stage_assign_results[stage][i]
                        novel_query_mask = assign_result.gt_inds > len(gt_bboxes[i])
                        novel_query_embedding.append(cls_score_feature_rkd[i, novel_query_mask])
                        novel_text_embedding.append(all_pseudo_box_clip_embedding[i][assign_result.labels[novel_query_mask] - num_base_classes])
                        novel_query_weight.append(all_weighting_score[i][assign_result.labels[novel_query_mask] - num_base_classes])
                    novel_query_embedding = F.normalize(torch.cat(novel_query_embedding), dim=-1)
                    novel_text_embedding = torch.cat(novel_text_embedding)
                    novel_query_weight = torch.cat(novel_query_weight)

                    padding_labels = torch.unique(torch.cat(idxs))
                    mask = torch.isin(padding_labels, total_unique_label)
                    padding_labels = padding_labels[~mask]
                    saved_centroid_embedding = F.normalize(self.class_centroid[stage].category_embeddings[self.class_centroid[stage].classid_to_idx[padding_labels]], dim=-1)

                    rkd_clip_embedding = torch.cat([self.category_embeddings[total_unique_label], novel_text_embedding, self.category_embeddings[padding_labels]])
                    rkd_query_embedding = torch.cat([normalize_query_text_embedding, novel_query_embedding, saved_centroid_embedding])
                    rkd_logits_clip = rkd_clip_embedding @ rkd_clip_embedding.t() * self.rkd_tau
                    rkd_logits_pred = rkd_query_embedding @ rkd_query_embedding.t() * self.rkd_tau
                    text_space_rkd_loss = text_space_rkd_loss + self.loss_text_rkd(
                        rkd_logits_pred,
                        rkd_logits_clip,)
                    
                    skd_query_embedding = torch.cat([normalize_query_text_embedding, novel_query_embedding])
                    skd_logits1 = skd_query_embedding @ rkd_clip_embedding.t() * self.skd_tau
                    # skd_logits2 = self.category_embeddings[total_unique_label] @ rkd_query_embedding.t() * self.skd_tau
                    text_space_skd_loss = text_space_skd_loss + self.loss_text_skd(
                        skd_logits1,
                        torch.arange(0, len(skd_query_embedding), device=total_unique_label.device, dtype=torch.long),
                        torch.cat([torch.ones((len(total_unique_label)), device=total_unique_label.device), novel_query_weight]))
                    # text_space_skd_loss = text_space_skd_loss + 0.5 * self.loss_text_skd(
                    #     skd_logits2,
                    #     torch.arange(0, len(total_unique_label), device=total_unique_label.device, dtype=torch.long),)
                    self.class_centroid[stage].update(total_unique_label, query_text_embedding)
                all_stage_loss[f'stage{stage}_loss_text_rkd'] = text_space_rkd_loss * self.stage_loss_weights[stage]
                all_stage_loss[f'stage{stage}_loss_text_skd'] = text_space_skd_loss * self.stage_loss_weights[stage]

                
            # standard detection loss
            bboxes_list = bbox_results['detach_bboxes_list']
            sampling_results = []
            # re-matching
            with torch.no_grad():
                for i in range(num_imgs):
                    assign_result = all_stage_assign_results[stage][i]
                    # remove low quality psuedo text label
                    # if len(all_pseudo_boxes[i]) > 0:
                    #     weighting_mask = all_weighting_score[i] < 0.4
                    #     class_inds = torch.nonzero(weighting_mask, as_tuple=True)[0] + len(gt_bboxes[i]) + 1
                    #     matched_query_inds = torch.any(assign_result.gt_inds.unsqueeze(1) == class_inds.unsqueeze(0), dim=-1)
                    #     assign_result.labels[matched_query_inds] = -1
                    #     assign_result.gt_inds[matched_query_inds] = 0
                    sampling_result = self.bbox_sampler[stage].sample(
                        assign_result, bboxes_list[i], gt_bboxes_with_pseudo_box[i])
                    sampling_results.append(sampling_result)
                    
            cls_score = torch.einsum('nqj,nmj->nqm', F.normalize(cls_score_feature, dim=-1), F.normalize(class_prompt, dim=-1)) * self.tau
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            objness = bbox_results['objness'].reshape(-1, 1)

            bbox_targets = self.bbox_head[stage].get_targets(
                    num_classes, num_base_classes, sampling_results, gt_bboxes_with_pseudo_box, gt_labels_with_pseudo_box, all_weighting_score, self.train_cfg[stage], True)

            single_stage_loss = self.bbox_head[stage].loss(
                num_classes,
                num_base_classes,
                cls_score.reshape(-1, cls_score.size(-1)),
                decode_bbox_pred.reshape(-1, 4),
                objness,
                *bbox_targets,
                avg_factor=avg_factor,
                imgs_whwh=imgs_whwh.flatten(0, 1))
            
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                    self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    x, 
                    img_no_normalize,
                    proposals,
                    query_content,
                    category_embeddings,
                    img_metas,
                    base_inds_tensor,
                    novel_inds_tensor,
                    imgs_whwh,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        query_xyzr = self.xyxy2xyzr(proposals).detach()
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        num_imgs = len(img_metas)
        bg_embedding = self.bg_embedding(torch.zeros((1,), device=query_content.device, dtype=torch.long)).unsqueeze(0).repeat(num_imgs, 1, 1)
        category_embeddings = torch.cat([category_embeddings, bg_embedding], dim=1)
        
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage, x, query_xyzr, query_content, img_metas)
            query_content = bbox_results['query_content']
            bboxes_list = bbox_results['detach_bboxes_list']
            query_xyzr = bbox_results['query_xyzr']
        cls_score_feature = bbox_results['cls_score_feature']
        if self.split_visual_text:
            cls_score_feature1 = cls_score_feature + self.visual2text(cls_score_feature)
        else:
            cls_score_feature1 = cls_score_feature
        cls_score = torch.einsum('nqj,nmj->nqm', F.normalize(cls_score_feature1, dim=-1), F.normalize(category_embeddings, dim=-1)) * self.tau
        cls_score = cls_score.softmax(-1)

        # ensemble
        if self.use_text_space_rkd_loss:
            #print(2)
            t = self.fc1(self.visual2text.layers[-1].weight.clone().detach())
            t_act = self.relu(t)
            transfer_weights = self.fc2(t_act)
            cls_score_feature2 = cls_score_feature + F.linear(cls_score_feature, weight=transfer_weights)
            cls_score2 = torch.einsum('nqj,nmj->nqm', F.normalize(cls_score_feature2, dim=-1), F.normalize(category_embeddings[:, :-1, :], dim=-1)) * self.skd_tau
            cls_score2 = cls_score2.softmax(-1)
        
            cls_score_ens = torch.zeros_like(cls_score)
            cls_score_ens[..., base_inds_tensor] = cls_score[..., base_inds_tensor] ** (1 - self.alpha) * cls_score2[..., base_inds_tensor] ** self.alpha
            cls_score_ens[..., novel_inds_tensor] = cls_score[..., novel_inds_tensor] ** (1 - self.beta) * cls_score2[..., novel_inds_tensor] ** self.beta
            cls_score_ens[..., -1] = cls_score[..., -1] # bg

            # Renormalize the probability to 1.
            cls_score = cls_score_ens / torch.sum(cls_score_ens, dim=-1, keepdim=True)

        if self.use_mask_head:
            rois = bbox2roi(bboxes_list)
            mask_results = self._mask_forward(stage, x[:self.mask_roi_extractor[stage].num_inputs], rois,
                                              bbox_results['attn_feats'])
            mask_results['mask_pred'] = mask_results['mask_pred'].reshape(
                num_imgs, -1, *mask_results['mask_pred'].size()[1:])

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            bboxes = bboxes_list[img_id]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bboxes /= bboxes.new_tensor(scale_factor)
            det_bbox, det_label, topk_indices = multiclass_nms(bboxes, cls_score_per_img,
                                                    0.005, dict(type='nms', iou_threshold=0.7),
                                                    self.test_cfg.max_per_img, return_inds=True)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

            # cls_score_per_img = cls_score_per_img[..., :-1]
            # scores_per_img, topk_indices = cls_score_per_img.flatten(
            #     0, 1).topk(
            #         self.test_cfg.max_per_img, sorted=False)
            # labels_per_img = topk_indices % num_classes
            # bbox_pred_per_img = bboxes[topk_indices // num_classes]
            # # if rescale:
            # #     scale_factor = img_metas[img_id]['scale_factor']
            # #     bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            # det_bboxes.append(
            #     torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            # det_labels.append(labels_per_img)
            # if self.image_id2 < 20:
            #     self.draw_dk_boxes(0, img_no_normalize[img_id], bbox_pred_per_img * bboxes.new_tensor(scale_factor))

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        if self.use_mask_head:
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            segm_results = []
            mask_pred = mask_results['mask_pred']
            for img_id in range(num_imgs):
                mask_pred_per_img = mask_pred[img_id].flatten(0, 1)[topk_indices // num_classes]
                mask_pred_per_img = mask_pred_per_img[:, None, ...]
                segm_result = self.mask_head[-1].get_seg_masks(
                    mask_pred_per_img, _bboxes[img_id], det_labels[img_id],
                    self.test_cfg, ori_shapes[img_id], scale_factors[img_id],
                    rescale)
                segm_results.append(segm_result)
            return list(zip(bbox_results, segm_results))
        else:
            return bbox_results



