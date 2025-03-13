# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, multiclass_nms
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


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
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 max_pseudo_box_num=5,
                 cls_tau=50,
                 skd_tau=20,
                 rkd_tau=5,
                 alpha=0.25,
                 beta=0.45,
                 text_dim=512,
                 pre_extracted_clip_text_feat='../ovd_resources/coco_proposals_text_embedding10/',
                 use_pseudo_box=True,
                 split_visual_text=True,
                 use_text_space_rkd_loss=True,
                 loss_visual_skd=dict(type='CrossEntropyLoss', loss_weight=0.5),
                 loss_visual_rkd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=5.0),
                 use_image_level_distill=True,
                 novel_obj_queue_dict=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(StandardRoIHead, self).__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, pretrained, init_cfg)
        
        self.max_pseudo_box_num = max_pseudo_box_num
        self.tau = cls_tau
        self.skd_tau = skd_tau
        self.rkd_tau = rkd_tau
        self.alpha = alpha
        self.beta = beta

        self.use_pseudo_box = use_pseudo_box
        self.bg_embedding = nn.Embedding(1, text_dim)

        if novel_obj_queue_dict is not None:
            self.queue = Queues(**novel_obj_queue_dict)

        self.split_visual_text = split_visual_text
        self.use_text_space_rkd_loss = use_text_space_rkd_loss
        self.fc = nn.Linear(1024, text_dim)
        if self.split_visual_text and self.use_text_space_rkd_loss:
            self.fc1 = nn.Linear(text_dim, text_dim*2)
            self.fc2 = nn.Linear(text_dim*2, text_dim)
            self.relu = nn.ReLU()
        
        if use_pseudo_box:
            self.loss_visual_skd = build_loss(loss_visual_skd)
            self.loss_visual_rkd = build_loss(loss_visual_rkd)
        self.use_pre_extracted_clip_text_feat = pre_extracted_clip_text_feat is not None
        self.pre_extracted_clip_text_feat_path = pre_extracted_clip_text_feat
        
        if self.use_text_space_rkd_loss:
            self.class_centroid = None
            self.category_embeddings = None
            self.loss_text_skd = build_loss(loss_visual_skd)
            self.loss_text_rkd = build_loss(loss_visual_rkd)
        self.use_image_level_distill = use_image_level_distill
        if self.use_image_level_distill:
            self.linear_transform = nn.Linear(256, text_dim)
            self.layernorm = nn.LayerNorm(256)
            self.loss_img_distill = build_loss(dict(type='CrossEntropyLoss', loss_weight=0.1))

            self.image_level_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[32])
            self.image_level_roi_extractor = build_roi_extractor(self.image_level_roi_extractor)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      img_no_normalize=None,
                      all_pseudo_boxes=None,
                      remapping_gt_labels=None,
                      category_embeddings=None,
                      idxs=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        num_imgs = len(img_metas)
        losses = dict()
        num_base_classes = category_embeddings.shape[1]
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
                        
                    
                    weighting_score = torch.sigmoid((torch.diag(pseudo_box_clip_image_feature @ pseudo_box_clip_text_feature.t()) - 0.31) / 0.026) * 1.3 # ViT-B-32 detpro
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
            img_distill_loss = torch.zeros(1, dtype=torch.float, device=x[0].device)
            image_box = [torch.tensor([[0, 0, img_meta['img_shape'][1], img_meta['img_shape'][0]]], dtype=gt_bboxes[0].dtype, device=gt_bboxes[0].device) for img_meta in img_metas]
            fused_x = x[3]
            for i in range(3):
                fused_x = fused_x + F.interpolate(x[i], fused_x.shape[-2:], mode='bilinear')
            image_box = bbox2roi(image_box)
            backbone_image_features = self.image_level_roi_extractor([fused_x], image_box).reshape(num_imgs, 256, 49)
            backbone_image_features = torch.mean(backbone_image_features, dim=-1)
            
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
            losses['img_distill_loss'] = img_distill_loss

 
        # expand class prompt
        expanded_class_prompt = []
        if self.use_pseudo_box:
            for i in range(num_imgs):
                pseudo_box_clip_embedding = all_pseudo_box_clip_embedding[i:] + all_pseudo_box_clip_embedding[:i]
                pseudo_box_clip_embedding = torch.cat(pseudo_box_clip_embedding)
                expanded_class_prompt.append(torch.cat([category_embeddings[i], pseudo_box_clip_embedding, self.bg_embedding(torch.zeros((1,), device=x[0].device, dtype=torch.long))]))
            num_classes = len(pseudo_box_clip_embedding) + num_base_classes
            prompt_len = [len(prompt) for prompt in expanded_class_prompt]
            prompt_len = min(prompt_len)
            expanded_class_prompt = [prompt[:prompt_len] for prompt in expanded_class_prompt]
        else:
            for i in range(num_imgs):
                expanded_class_prompt.append(torch.cat([category_embeddings[i], self.bg_embedding(torch.zeros((1,), device=x[0].device, dtype=torch.long)),]))
            num_classes = num_base_classes

        class_prompt = torch.stack(expanded_class_prompt)
        losses['num_pseudo_box'] = torch.tensor(num_classes - num_base_classes, dtype=torch.float, device=x[0].device) / num_imgs


        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes_with_pseudo_box[i], gt_bboxes_ignore[i],
                    gt_labels_with_pseudo_box[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes_with_pseudo_box[i],
                    gt_labels_with_pseudo_box[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)


        # bbox head forward and loss
        if self.with_bbox:
            labels, label_weights, bbox_targets, bbox_weights = self.bbox_head.get_targets(sampling_results, gt_bboxes_with_pseudo_box,
                                                    gt_labels_with_pseudo_box, self.train_cfg, num_classes)
            roi_nums = [len(res.bboxes) for res in sampling_results]
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results, roi_feats = self._bbox_forward(x, rois)
            roi_feats = self.fc(roi_feats)
            if self.split_visual_text and self.use_text_space_rkd_loss:
                t = self.fc1(roi_feats)
                t_act = self.relu(t)
                cls_score_feature_rkd = roi_feats + self.fc2(t_act)
            else:
                cls_score_feature_rkd = roi_feats

            roi_feats = torch.split(roi_feats, roi_nums, dim=0)

            if self.use_pseudo_box:
                all_query_embedding = []
                all_clip_embedding = []
                for i in range(num_imgs):
                    clip_image_feature_i = torch.cat([gt_bboxes_clip_image_feature[i], all_pseudo_boxes_clip_image_feature[i]])
                    query_embedding = roi_feats[i][:len(sampling_results[i].pos_inds)]
                    for j in range(len(gt_bboxes_with_pseudo_box[i])):
                        query_embedding_per_gt = query_embedding[sampling_results[i].pos_assigned_gt_inds == j]
                        if len(query_embedding_per_gt) > 0:
                            all_query_embedding.append(F.normalize(torch.mean(query_embedding_per_gt, dim=0, keepdim=True)))
                            all_clip_embedding.append(clip_image_feature_i[j:j+1])
                
                if len(all_clip_embedding) > 0:
                    all_clip_embedding = torch.cat(all_clip_embedding)
                    all_query_embedding = torch.cat(all_query_embedding)

                loss_rkd = torch.zeros(1, dtype=torch.float, device=x[0].device)
                loss_skd = torch.zeros(1, dtype=torch.float, device=x[0].device)
                if len(all_clip_embedding) >= 2:
                    saved_obj_query = self.queue.get_queue('obj_query')
                    saved_clip_embedding = self.queue.get_queue('clip_query')
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
                    self.queue.dequeue_and_enqueue({'obj_query': all_query_embedding.detach()})
                    self.queue.dequeue_and_enqueue({'clip_query': all_clip_embedding.detach()})

                losses['loss_skd'] = loss_skd
                losses['loss_rkd'] = loss_rkd


            # text space distillaton loss
            if self.use_text_space_rkd_loss:
                text_space_rkd_loss = torch.zeros(1, dtype=torch.float, device=x[0].device)
                text_space_skd_loss = torch.zeros(1, dtype=torch.float, device=x[0].device)
                total_unique_label = torch.unique(torch.cat(gt_labels))
                if len(total_unique_label) >= 2:
                    query_text_embedding = []
                    # base class query
                    for label_id in total_unique_label:
                        class_mask = labels == label_id
                        matched_query = cls_score_feature_rkd[class_mask]
                        fused_class_query = torch.mean(matched_query, dim=0)
                        query_text_embedding.append(fused_class_query)
                    query_text_embedding = torch.stack(query_text_embedding, dim=0)
                    normalize_query_text_embedding = F.normalize(query_text_embedding, dim=-1)

                    # novel class query
                    # novel_query_embedding = []
                    # novel_text_embedding = []
                    # novel_query_weight = []
                    # for i in range(num_imgs):
                    #     assign_result = all_stage_assign_results[stage][i]
                    #     novel_query_mask = assign_result.gt_inds > len(gt_bboxes[i])
                    #     novel_query_embedding.append(cls_score_feature_rkd[i, novel_query_mask])
                    #     novel_text_embedding.append(all_pseudo_box_clip_embedding[i][assign_result.labels[novel_query_mask] - num_base_classes])
                    #     novel_query_weight.append(all_weighting_score[i][assign_result.labels[novel_query_mask] - num_base_classes])
                    # novel_query_embedding = F.normalize(torch.cat(novel_query_embedding), dim=-1)
                    # novel_text_embedding = torch.cat(novel_text_embedding)
                    # novel_query_weight = torch.cat(novel_query_weight)

                    padding_labels = torch.unique(torch.cat(idxs))
                    mask = torch.isin(padding_labels, total_unique_label)
                    padding_labels = padding_labels[~mask]
                    saved_centroid_embedding = F.normalize(self.class_centroid.category_embeddings[self.class_centroid.classid_to_idx[padding_labels]], dim=-1)

                    rkd_clip_embedding = torch.cat([self.category_embeddings[total_unique_label], self.category_embeddings[padding_labels]])
                    rkd_query_embedding = torch.cat([normalize_query_text_embedding, saved_centroid_embedding])
                    rkd_logits_clip = rkd_clip_embedding @ rkd_clip_embedding.t() * self.rkd_tau
                    rkd_logits_pred = rkd_query_embedding @ rkd_query_embedding.t() * self.rkd_tau
                    text_space_rkd_loss = text_space_rkd_loss + self.loss_text_rkd(
                        rkd_logits_pred,
                        rkd_logits_clip,)
                    
                    skd_query_embedding = normalize_query_text_embedding
                    skd_logits1 = skd_query_embedding @ rkd_clip_embedding.t() * self.skd_tau
                    text_space_skd_loss = text_space_skd_loss + self.loss_text_skd(
                        skd_logits1,
                        torch.arange(0, len(skd_query_embedding), device=total_unique_label.device, dtype=torch.long),)
                    # text_space_skd_loss = text_space_skd_loss + 0.5 * self.loss_text_skd(
                    #     skd_logits2,
                    #     torch.arange(0, len(total_unique_label), device=total_unique_label.device, dtype=torch.long),)
                    self.class_centroid.update(total_unique_label, query_text_embedding)
                losses['loss_text_rkd'] = text_space_rkd_loss
                losses['loss_text_skd'] = text_space_skd_loss
            
            
            bbox_weights[labels >= num_base_classes] = 0 # pseudo box do not use for box supervision
            # use label weighting
            if num_classes > num_base_classes:
                labels_splits = torch.split(labels, roi_nums, dim=0)
                label_weights_splits = torch.split(label_weights, roi_nums, dim=0)
                new_label_weights_splits = []
                for i, label_weights in enumerate(label_weights_splits):
                    pos_gt_labels = labels_splits[i][:len(sampling_results[i].pos_inds)]
                    pos_inds = torch.arange(0, len(sampling_results[i].pos_inds), device=pos_gt_labels.device)
                    pos_inds_pseudo = pos_inds[pos_gt_labels >= num_base_classes]
                    label_inds = pos_gt_labels[pos_gt_labels >= num_base_classes] - num_base_classes
                    label_weights[pos_inds_pseudo] = all_weighting_score[i][label_inds]
                    new_label_weights_splits.append(label_weights)
                label_weights = torch.cat(new_label_weights_splits)
            
            cls_feats = torch.split(bbox_results['cls_score'], roi_nums, dim=0)
            cls_scores = []
            for i, cls_feat in enumerate(cls_feats):
                cls_score = torch.einsum('qj,mj->qm', F.normalize(cls_feat, dim=-1), F.normalize(class_prompt[i], dim=-1)) * self.tau
                cls_scores.append(cls_score)

            loss_bbox = self.bbox_head.loss(torch.cat(cls_scores),
                                            bbox_results['bbox_pred'], rois,
                                            labels, label_weights, bbox_targets, bbox_weights)

            bbox_results.update(loss_bbox=loss_bbox)
            losses.update(bbox_results['loss_bbox'])


        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, roi_feats = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, roi_feats


    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    img_no_normalize,
                    class_prompt,
                    base_inds_tensor,
                    novel_inds_tensor,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        proposals = proposal_list
        rcnn_test_cfg = self.test_cfg
        num_imgs = len(img_metas)
        bg_embedding = self.bg_embedding(torch.zeros((1,), device=x[0].device, dtype=torch.long)).unsqueeze(0).repeat(num_imgs, 1, 1)
        category_embeddings = torch.cat([class_prompt, bg_embedding], dim=1)

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results, roi_feats = self._bbox_forward(x, rois)
        roi_feats = self.fc(roi_feats)

        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_feats = cls_score.split(num_proposals_per_img, 0)
        roi_feats = roi_feats.split(num_proposals_per_img, 0)
        # if self.use_text_space_rkd_loss:
        #     t = self.fc1(roi_feats)
        #     t_act = self.relu(t)
        #     cls_score_feature_rkd = roi_feats + self.fc2(t_act)
        #     cls_score_feature_rkd = cls_score_feature_rkd.split(num_proposals_per_img, 0)
        cls_scores = []
        for i, cls_feat in enumerate(cls_feats):
            cls_score = torch.einsum('qj,mj->qm', F.normalize(cls_feat, dim=-1), F.normalize(category_embeddings[i], dim=-1)) * self.tau
            cls_score = torch.softmax(cls_score, dim=-1)
            # ensemble
            if self.use_pseudo_box:
                # cls_score_feature_rkd_i = cls_score_feature_rkd[i]
                cls_score2 = torch.einsum('qj,mj->qm', F.normalize(roi_feats[i], dim=-1), F.normalize(category_embeddings[i, :-1, :], dim=-1)) * self.skd_tau
                cls_score2 = cls_score2.sigmoid()
            
                cls_score_ens = torch.zeros_like(cls_score)
                cls_score_ens[..., base_inds_tensor] = cls_score[..., base_inds_tensor] ** (1 - self.alpha) * cls_score2[..., base_inds_tensor] ** self.alpha
                cls_score_ens[..., novel_inds_tensor] = cls_score[..., novel_inds_tensor] ** (1 - self.beta) * cls_score2[..., novel_inds_tensor] ** self.beta
                cls_score_ens[..., -1] = cls_score[..., -1] # bg

                # Renormalize the probability to 1.
                cls_score = cls_score_ens / torch.sum(cls_score_ens, dim=-1, keepdim=True)
            cls_scores.append(cls_score)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_scores[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                
                # bbox_pred would be None in some detector when with_reg is False,
                # e.g. Grid R-CNN.
                if bbox_pred[i] is not None:
                    bboxes = self.bbox_head.bbox_coder.decode(
                        rois[i][..., 1:], bbox_pred[i], max_shape=img_shapes[i])
                else:
                    bboxes = rois[i][:, 1:].clone()
                    if img_shapes[i] is not None:
                        bboxes[:, [0, 2]].clamp_(min=0, max=img_shapes[i][1])
                        bboxes[:, [1, 3]].clamp_(min=0, max=img_shapes[i][0])

                if rescale and bboxes.size(0) > 0:
                    scale_factor = bboxes.new_tensor(scale_factors[i])
                    bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                        bboxes.size()[0], -1)

                if rcnn_test_cfg is None:
                    det_bbox = bboxes
                    det_label = cls_scores[i]
                else:
                    det_bbox, det_label = multiclass_nms(bboxes, cls_scores[i],
                                                            rcnn_test_cfg.score_thr, rcnn_test_cfg.nms,
                                                            rcnn_test_cfg.max_per_img)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
