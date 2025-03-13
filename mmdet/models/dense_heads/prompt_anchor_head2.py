# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv import ConfigDict
from mmcv.ops import batched_nms
from mmcv.cnn import normal_init, bias_init_with_prob, ConvModule

from mmdet.core import (anchor_inside_flags, build_assigner, distance2bbox, reduce_mean,
                        build_sampler, multi_apply, MlvlPointGenerator, unmap, bbox_cxcywh_to_xyxy)
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.models.roi_heads.bbox_heads.sampling_3d_operator import sampling_3d
from ..builder import build_loss, HEADS
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi


class AuxLoss(nn.Module):
    def __init__(
        self,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(assigner=dict(type='TopkHungarianAssigner', topk=8),
                       alpha=1,
                       beta=6),
    ):
        super(AuxLoss, self).__init__()
        self.train_cfg = train_cfg
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.assigner = build_assigner(self.train_cfg['assigner'])

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, alignment_metrics):

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, alignment_metrics)
        cls_loss_func = self.loss_cls

        loss_cls = cls_loss_func(cls_score,
                                 targets,
                                 label_weights,
                                 avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = cls_score.size(-1)
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = alignment_metrics[pos_inds]

            loss_bbox = self.loss_bbox(pos_decode_bbox_pred,
                                       pos_decode_bbox_targets,
                                       weight=pos_bbox_weight,
                                       avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def __call__(self, type, cls_scores, bbox_preds, priors=None, valid_flags=None, gt_bboxes=None, gt_labels=None, img_metas=None,
                 **kwargs):
        # 0: main, 1: aux
        flatten_cls_scores = cls_scores
        flatten_bbox_preds = bbox_preds

        cls_reg_targets = self.get_targets(
            type,
            flatten_cls_scores,
            flatten_bbox_preds,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels,
            priors=priors,
            valid_flags=valid_flags,
        )
        (labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                flatten_cls_scores,
                flatten_bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(self,
                    type,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    img_metas,
                    gt_labels_list=None,
                    priors=None, valid_flags=None,
                    **kwargs):

        if type == 0: # main
            (all_labels, all_label_weights, all_bbox_targets,
            all_assign_metrics) = multi_apply(self._get_target_single, cls_scores,
                                            bbox_preds, gt_bboxes_list,
                                            gt_labels_list, img_metas)
        else:
            (all_labels, all_label_weights, all_bbox_targets,
            all_assign_metrics) = multi_apply(self._get_aux_target_single, cls_scores,
                                            bbox_preds, priors, valid_flags, gt_bboxes_list,
                                            gt_labels_list, img_metas)

        return (all_labels, all_label_weights, all_bbox_targets,
                all_assign_metrics)

    def _get_target_single(self, cls_scores, bbox_preds, gt_bboxes, gt_labels,
                           img_meta, **kwargs):
        num_gt = len(gt_labels)
        if num_gt == 0:
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                                 dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                          dtype=torch.float)

            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        assign_result = self.assigner.assign(cls_scores, bbox_preds, gt_bboxes,
                                             gt_labels, img_meta,
                                             self.train_cfg.get('alpha', 1),
                                             self.train_cfg.get('beta', 6))
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, bbox_preds,
                                              gt_bboxes)

        num_valid_anchors = len(cls_scores)
        bbox_targets = torch.zeros_like(bbox_preds)
        labels = bbox_preds.new_full((num_valid_anchors, ),
                                     cls_scores.size(-1),
                                     dtype=torch.long)
        label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                             dtype=torch.float)
        norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                      dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only dense_heads gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = sampling_result.pos_assigned_gt_inds == gt_inds
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[
                pos_inds[gt_class_inds]] = pos_norm_alignment_metrics

        return (labels, label_weights, bbox_targets, norm_alignment_metrics)

    def _get_aux_target_single(self,
                           cls_scores,
                           bbox_preds,
                           flat_anchors,
                           valid_flags,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           True)
        num_gt = len(gt_labels)
        if (num_gt == 0) or (not inside_flags.any()):
            num_valid_anchors = len(cls_scores)
            bbox_targets = torch.zeros_like(bbox_preds)
            labels = bbox_preds.new_full((num_valid_anchors, ),
                                         cls_scores.size(-1),
                                         dtype=torch.long)
            label_weights = bbox_preds.new_zeros(num_valid_anchors,
                                                 dtype=torch.float)
            norm_alignment_metrics = bbox_preds.new_zeros(num_valid_anchors,
                                                          dtype=torch.float)

            return (labels, label_weights, bbox_targets,
                    norm_alignment_metrics)

        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        anchors = bbox_cxcywh_to_xyxy(anchors)
        assign_result = self.assigner.assign(
            cls_scores[inside_flags, :], bbox_preds[inside_flags, :], anchors,
            gt_bboxes, None, gt_labels, self.train_cfg.get('alpha', 1), self.train_cfg.get('beta', 6))
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  cls_scores.size(-1),
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
                
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds == gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=cls_scores.size(-1))
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
            
        return (labels, label_weights, bbox_targets, norm_alignment_metrics)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class StaRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(self, hidden_dim, num_levels, clip_obj_embedding_file):
        super().__init__()
        # 3x3 conv for the hidden representation
        in_channels = hidden_dim
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj_embeddings = torch.load(clip_obj_embedding_file, 'cpu').float()
        self.obj_embeddings = F.normalize(obj_embeddings, dim=1).to(device)
        
        # 1x1 conv for predicting objectness logits
        self.anchor_deltas = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 4, kernel_size=1, stride=1),
        )

        self.objectness_logits = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.obj_embeddings.shape[-1], kernel_size=1, stride=1),
        )

        self.init_weights()

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(num_levels)])

        self.obj_scale = nn.Parameter(torch.Tensor([1.0]))
        bias_cls = bias_init_with_prob(0.01)
        self.obj_bias = nn.Parameter(torch.Tensor([bias_cls])) # -4.59511985013459
        
    
    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        
        # normal_init(self.anchor_deltas[-1], std=0.01)
        # bias_init = bias_init_with_prob(0.01)
        # nn.init.constant_(self.objectness_logits[-1].bias, bias_init)
        

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for level, x in enumerate(features):
            N, C, H, W = x.shape
            pred_anchor_deltas.append(self.scales[level](self.anchor_deltas(x).exp()).float())
            objectness_feat = self.objectness_logits(x).flatten(2).permute(0, 2, 1)
            objectnesses = self.obj_scale * (F.normalize(objectness_feat) @ F.normalize(self.obj_embeddings).unsqueeze(-1)) + self.obj_bias
            pred_objectness_logits.append(objectnesses.permute(0, 2, 1).reshape(N, 1, H, W).sigmoid())

        return pred_objectness_logits, pred_anchor_deltas


@HEADS.register_module()
class PromptAnchorHead2(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                in_channels,
                strides,
                clip_obj_embedding_file,
                feat_channels=256,
                aux_loss=ConfigDict(
                    loss_cls=dict(
                        type='QualityFocalLoss',
                        use_sigmoid=True,
                        activated=True,  # use probability instead of logit as input
                        beta=2.0,
                        loss_weight=1.0),
                    loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                    train_cfg=dict(assigner=dict(type='TaskAlignedAssigner',
                                                topk=8),
                                alpha=1,
                                beta=6),
                ),
                main_loss=ConfigDict(
                    loss_cls=dict(
                        type='QualityFocalLoss',
                        use_sigmoid=True,
                        activated=True,  # use probability instead of logit as input
                        beta=2.0,
                        loss_weight=1.0),
                    loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                    train_cfg=dict(assigner=dict(type='TopkHungarianAssigner',
                                                topk=1),
                                alpha=1,
                                beta=6),
                ),
                shuffle_channles=64,
                dqs_cfg=dict(type='nms', iou_threshold=0.7, nms_pre=1000),
                offset=0.5,
                num_query=300,
                rpn_loss_coef=1,
                train_cfg=None,
                test_cfg=None,
                init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(PromptAnchorHead2, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_query = num_query
        self.token_norm = nn.LayerNorm(in_channels)

        self.aux_loss = AuxLoss(**aux_loss)
        self.main_loss = AuxLoss(**main_loss)

        self.prior_generator = MlvlPointGenerator(strides, offset=offset)
        self.num_levels = len(strides)
        self.main_head = StaRPNHead(self.in_channels, self.num_levels, clip_obj_embedding_file)
        self.aux_head = StaRPNHead(self.in_channels, self.num_levels, clip_obj_embedding_file)
        self.strides = strides
        self.fuse_lvl_list = []
        for lvl in range(self.num_levels):
            top_lvl = min(lvl + 1, self.num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            tar_lvl = lvl
            self.fuse_lvl_list.append((tar_lvl, top_lvl, dow_lvl))
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.conv1 = ConvModule(in_channels, in_channels, 3,
                           stride=1,
                           padding=3 // 2,
                           norm_cfg=self.norm_cfg)
        self.conv2 = ConvModule(in_channels, in_channels, 3,
                           stride=1,
                           padding=3 // 2,
                           norm_cfg=self.norm_cfg)
        self.shuffle_channles = shuffle_channles
        self.remain_chs = in_channels - self.shuffle_channles * 2
        self.dqs_cfg = dqs_cfg
        self.rpn_loss_coef = rpn_loss_coef


    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'for consistency or also use '
                      '`num_base_priors` instead')
        return self.prior_generator.num_base_priors[0]

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
                      'please use "prior_generator" instead')
        return self.prior_generator
    
    def _single_shuffle(self, inputs, fuse_lvl_list):

        fused_inputs = []
        for fuse_lvl_tuple in fuse_lvl_list:
            tar_lvl, top_lvl, dow_lvl = fuse_lvl_tuple
            tar_input = inputs[tar_lvl]
            top_input = inputs[top_lvl]
            down_input = inputs[dow_lvl]
            remain = tar_input[:, :self.remain_chs]
            from_top = top_input[:, self.remain_chs:][:, self.shuffle_channles:]
            from_top = F.interpolate(from_top,
                                        size=tar_input.shape[-2:],
                                        mode='bilinear',
                                        align_corners=True)
            from_down = down_input[:, self.remain_chs:][:, :self.shuffle_channles]
            from_down = F.interpolate(from_down,
                                        size=tar_input.shape[-2:],
                                        mode='bilinear',
                                        align_corners=True)
            fused_inputs.append(
                torch.cat([remain, from_top, from_down], dim=1))

        return fused_inputs


    def forward_train(self,
                      x,
                      category_embeddings,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        gt_labels = [gt_label.clone() * 0 for gt_label in gt_labels] # class-agnostic

        head_features = x[-self.num_levels:]
        head_features = [self.conv1(x_i) for x_i in head_features]
        head_features = self._single_shuffle(head_features, self.fuse_lvl_list)
        head_features = [self.conv2(x_i) for x_i in head_features]

        pred_objectness_logits_main, pred_anchor_deltas_main = self.main_head(head_features)
        pred_objectness_logits_aux, pred_anchor_deltas_aux = self.aux_head(head_features)
        main_results = dict(cls_scores_list=pred_objectness_logits_main,
                            bbox_preds_list=pred_anchor_deltas_main,)
        aux_results = dict(cls_scores_list=pred_objectness_logits_aux,
                            bbox_preds_list=pred_anchor_deltas_aux,)

        main_loss_inputs, aux_loss_inputs = self.get_inputs(
            main_results, aux_results, img_metas=img_metas)

        losses = dict()
        aux_loss = self.aux_loss(1, *aux_loss_inputs,
                                 gt_bboxes=gt_bboxes,
                                 gt_labels=gt_labels,
                                 img_metas=img_metas)
        for k, v in aux_loss.items():
            losses[f'aux_{k}'] = v

        main_loss = self.main_loss(0, *main_loss_inputs,
                                   gt_bboxes=gt_bboxes,
                                   gt_labels=gt_labels,
                                   img_metas=img_metas)
        losses.update(main_loss)

        for k, v in losses.items():
            losses[k] = sum(v) * self.rpn_loss_coef

        xyzr = []
        proposals = []
        proposals_scores = []
        for img_id in range(len(img_metas)):
            singl_scores = main_loss_inputs[0][img_id]
            singl_bboxes = main_loss_inputs[1][img_id]
            singl_bboxes = singl_bboxes.detach()

            select_ids = torch.sort(singl_scores[:, 0],
                                    descending=True).indices[:self.num_query]
            singl_bboxes = singl_bboxes[select_ids]
            proposals.append(singl_bboxes)
            singl_scores = singl_scores[:, 0][select_ids]
            proposals_scores.append(singl_scores)
            xy = 0.5 * (singl_bboxes[..., 0:2] + singl_bboxes[..., 2:4])
            wh = singl_bboxes[..., 2:4] - singl_bboxes[..., 0:2]
            z = (wh).prod(-1, keepdim=True).sqrt().log2()
            r = (wh[..., 1:2]/wh[..., 0:1]).log2()
            xyzr_i = torch.cat([xy, z, r], dim=-1)
            xyzr.append(xyzr_i)
        xyzr = torch.stack(xyzr, dim=0)
        proposals = torch.stack(proposals, dim=0)
        proposals_scores = torch.stack(proposals_scores, dim=0).detach()

        sample_points = xyzr[..., :3].unsqueeze(2).unsqueeze(2)
        proposal_feats = sampling_3d(sample_points, x[:4], [4, 8, 16, 32])[0].squeeze(2).squeeze(2)

        proposal_feats = self.token_norm(proposal_feats)

        return losses, proposals, proposals_scores, proposal_feats, imgs_whwh

    def simple_test_rpn(self, x, category_embeddings, img_metas):
        """Test function without test-time augmentation.

        Args:
            x (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        head_features = x[-self.num_levels:]
        head_features = [self.conv1(x_i) for x_i in head_features]
        head_features = self._single_shuffle(head_features, self.fuse_lvl_list)
        head_features = [self.conv2(x_i) for x_i in head_features]

        pred_objectness_logits_main, pred_anchor_deltas_main = self.main_head(head_features)
        main_results = dict(cls_scores_list=pred_objectness_logits_main,
                            bbox_preds_list=pred_anchor_deltas_main,)

        main_loss_inputs, aux_loss_inputs = self.get_inputs(
            main_results, None, img_metas=img_metas)

        xyzr = []
        proposals = []
        proposals_scores = []
        for img_id in range(len(img_metas)):
            singl_scores = main_loss_inputs[0][img_id]
            singl_bboxes = main_loss_inputs[1][img_id]
            singl_bboxes = singl_bboxes.detach()

            select_ids = torch.sort(singl_scores[:, 0],
                                    descending=True).indices[:self.num_query]
            singl_bboxes = singl_bboxes[select_ids]
            proposals.append(singl_bboxes)
            singl_scores = singl_scores[:, 0][select_ids]
            proposals_scores.append(singl_scores)
            xy = 0.5 * (singl_bboxes[..., 0:2] + singl_bboxes[..., 2:4])
            wh = singl_bboxes[..., 2:4] - singl_bboxes[..., 0:2]
            z = (wh).prod(-1, keepdim=True).sqrt().log2()
            r = (wh[..., 1:2]/wh[..., 0:1]).log2()
            xyzr_i = torch.cat([xy, z, r], dim=-1)
            xyzr.append(xyzr_i)
        xyzr = torch.stack(xyzr, dim=0)
        proposals = torch.stack(proposals, dim=0)
        proposals_scores = torch.stack(proposals_scores, dim=0).detach()

        sample_points = xyzr[..., :3].unsqueeze(2).unsqueeze(2)
        proposal_feats = sampling_3d(sample_points, x[:4], [4, 8, 16, 32])[0].squeeze(2).squeeze(2)

        proposal_feats = self.token_norm(proposal_feats)

        return proposals, proposals_scores, proposal_feats, imgs_whwh

    
    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        # anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(torch.cat(multi_level_flags))

        return multi_level_anchors, valid_flag_list
    
    def get_inputs(self, main_results, aux_results, img_metas=None):

        mlvl_score = main_results['cls_scores_list']
        num_levels = len(mlvl_score)
        featmap_sizes = [mlvl_score[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, mlvl_score[0].device)

        all_cls_scores, all_bbox_preds, all_query_ids = self.pre_dqs(
            **main_results, mlvl_priors=mlvl_priors, img_metas=img_metas)
        # test stage
        if aux_results is None:
            (aux_cls_scores, aux_bbox_preds) = (None, None)
        else:
            aux_cls_scores, aux_bbox_preds, all_priors = self.aux_pre_dps(
                **aux_results, mlvl_priors=mlvl_priors, img_metas=img_metas)

        nms_all_cls_scores, nms_all_bbox_preds = self.dqs(
            all_cls_scores, all_bbox_preds)

        if aux_results is None:
            return (nms_all_cls_scores, nms_all_bbox_preds), None
        else:
            return (nms_all_cls_scores, nms_all_bbox_preds), (aux_cls_scores, aux_bbox_preds, all_priors, valid_flag_list)

    def dqs(self, all_mlvl_scores, all_mlvl_bboxes):
        ddq_bboxes = []
        ddq_scores = []
        for mlvl_bboxes, mlvl_scores in zip(all_mlvl_bboxes, all_mlvl_scores):
            if mlvl_bboxes.numel() == 0:
                return mlvl_bboxes, mlvl_scores

            det_bboxes, ddq_idxs = batched_nms(mlvl_bboxes,
                                               mlvl_scores[:, 0],
                                               torch.ones(len(mlvl_scores)),
                                               self.dqs_cfg)

            ddq_bboxes.append(mlvl_bboxes[ddq_idxs])
            ddq_scores.append(mlvl_scores[ddq_idxs])
        return ddq_scores, ddq_bboxes

    def pre_dqs(self,
                cls_scores_list=None,
                bbox_preds_list=None,
                mlvl_priors=None,
                img_metas=None,
                **kwargs):

        num_imgs = cls_scores_list[0].size(0)
        all_cls_scores = []
        all_bbox_preds = []
        all_query_ids = []
        for img_id in range(num_imgs):

            single_cls_score_list = select_single_mlvl(cls_scores_list,
                                                       img_id,
                                                       detach=False)
            sinlge_bbox_pred_list = select_single_mlvl(bbox_preds_list,
                                                       img_id,
                                                       detach=False)
            cls_score, bbox_pred, query_inds = self._get_topk(
                single_cls_score_list, sinlge_bbox_pred_list, mlvl_priors,
                img_metas[img_id])
            all_cls_scores.append(cls_score)
            all_bbox_preds.append(bbox_pred)
            all_query_ids.append(query_inds)
        return all_cls_scores, all_bbox_preds, all_query_ids

    def _get_topk(self, cls_score_list, bbox_pred_list, mlvl_priors, img_meta,
                  **kwargs):
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_query_inds = []
        start_inds = 0
        for level_idx, (cls_score, bbox_pred, priors, stride) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                     mlvl_priors, \
                        self.prior_generator.strides)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, 1)
            if self.dqs_cfg:
                nms_pre = self.dqs_cfg.pop('nms_pre', 1000)
            else:
                if self.training:
                    nms_pre = len(cls_score)
                else:
                    nms_pre = 1000
            results = filter_scores_and_topk(
                cls_score, 0, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, cls_score=cls_score))
            scores, labels, keep_idxs, filtered_results = results
            keep_idxs = keep_idxs + start_inds
            start_inds = start_inds + len(cls_score)
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            cls_score = filtered_results['cls_score']
            bbox_pred = bbox_pred * stride[0]
            bbox_pred = distance2bbox(priors, bbox_pred)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(cls_score)
            mlvl_query_inds.append(keep_idxs)

        return torch.cat(mlvl_scores), torch.cat(mlvl_bboxes), torch.cat(
            mlvl_query_inds)
    
    def aux_pre_dps(self,
                cls_scores_list=None,
                bbox_preds_list=None,
                mlvl_priors=None,
                img_metas=None,
                **kwargs):
        num_imgs = cls_scores_list[0].size(0)
        all_cls_scores = []
        all_bbox_preds = []
        all_priors = []
        for img_id in range(num_imgs):

            single_cls_score_list = select_single_mlvl(cls_scores_list,
                                                       img_id,
                                                       detach=False)
            sinlge_bbox_pred_list = select_single_mlvl(bbox_preds_list,
                                                       img_id,
                                                       detach=False)
            mlvl_bboxes = []
            mlvl_scores = []
            mlvl_priors_list = []
            for cls_score, bbox_pred, priors, stride in \
                    zip(single_cls_score_list, sinlge_bbox_pred_list, mlvl_priors, self.prior_generator.strides):

                assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                cls_score = cls_score.permute(1, 2, 0).reshape(-1, 1)
                
                bbox_pred = bbox_pred * stride[0]
                bbox_pred = distance2bbox(priors, bbox_pred)
                mlvl_bboxes.append(bbox_pred)
                mlvl_scores.append(cls_score)
                mlvl_priors_list.append(priors)

            all_cls_scores.append(torch.cat(mlvl_scores))
            all_bbox_preds.append(torch.cat(mlvl_bboxes))
            all_priors.append(torch.cat(mlvl_priors_list))
        return all_cls_scores, all_bbox_preds, all_priors

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """

        pass

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        """
        pass


