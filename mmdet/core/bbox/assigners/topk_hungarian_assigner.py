import torch
from ..transforms import (bbox_xyxy_to_cxcywh,)
from .assign_result import AssignResult
from .task_aligned_assigner import TaskAlignedAssigner
from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from scipy.optimize import linear_sum_assignment


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

@BBOX_ASSIGNERS.register_module()
class TopkHungarianAssigner(TaskAlignedAssigner):
    def __init__(self,
                 *args,
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                 **kwargs):
        super(TopkHungarianAssigner, self).__init__(*args, **kwargs)

        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self,
               pred_scores,
               decode_bboxes,
               gt_bboxes,
               gt_labels,
               img_meta,
               alpha=1,
               beta=6,
               **kwargs):
        pred_scores = pred_scores.detach()
        decode_bboxes = decode_bboxes.detach()
        temp_overlaps = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        alignment_metrics = bbox_scores**alpha * temp_overlaps**beta

        # all cost
        h, w, _ = img_meta['img_shape']
        img_whwh = pred_scores.new_tensor([w, h, w, h])
        normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(decode_bboxes / img_whwh)
        normalize_gt_bboxes = gt_bboxes / img_whwh
        reg_cost = self.reg_cost(normalize_bbox_ccwh, normalize_gt_bboxes)
        iou_cost = self.iou_cost(decode_bboxes, gt_bboxes)
        cls_cost = self.cls_cost(inverse_sigmoid(pred_scores), gt_labels)
        all_cost = cls_cost + reg_cost + iou_cost

        num_gt, num_bboxes = gt_bboxes.size(0), pred_scores.size(0)
        if num_gt > 0:
            # assign 0 by default
            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    0,
                                                    dtype=torch.long)
            select_cost = all_cost
            # num anchor * (num_gt * topk)
            topk = min(self.topk, int(len(select_cost) / num_gt))
            # num_anchors * (num_gt * topk)
            repeat_select_cost = select_cost[...,
                                             None].repeat(1, 1, topk).view(
                                                 select_cost.size(0), -1)
            # anchor index and gt index
            matched_row_inds, matched_col_inds = linear_sum_assignment(
                repeat_select_cost.detach().cpu().numpy())
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                pred_scores.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                pred_scores.device)

            match_gt_ids = matched_col_inds // topk
            candidate_idxs = matched_row_inds

            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)

            if candidate_idxs.numel() > 0:
                assigned_labels[candidate_idxs] = gt_labels[match_gt_ids]
            else:
                assigned_labels = None

            assigned_gt_inds[candidate_idxs] = match_gt_ids + 1

            overlaps = self.iou_calculator(decode_bboxes[candidate_idxs],
                                           gt_bboxes[match_gt_ids],
                                           is_aligned=True).detach()

            temp_pos_alignment_metrics = alignment_metrics[candidate_idxs]
            pos_alignment_metrics = torch.gather(temp_pos_alignment_metrics, 1,
                                                 match_gt_ids[:,
                                                              None]).view(-1)
            assign_result = AssignResult(num_gt,
                                         assigned_gt_inds,
                                         overlaps,
                                         labels=assigned_labels)

            assign_result.assign_metrics = pos_alignment_metrics
            return assign_result
        else:

            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)

            assigned_labels = pred_scores.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)

            assigned_gt_inds[:] = 0
            return AssignResult(0,
                                assigned_gt_inds,
                                None,
                                labels=assigned_labels)
