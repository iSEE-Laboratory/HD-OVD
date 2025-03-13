# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


# def calculate_loss_func(pred,
#                        target,
#                        beta=2.0,
#                        weight=None,
#                        reduction='mean',
#                        avg_factor=None):
    
#     bg_class_ind = pred.shape[-1] # bg
#     pos = ((target >= 0) & (target < bg_class_ind)).nonzero().squeeze(1)
#     pos_label = target[pos].long()
#     soft_label = pred.new_ones(pred.shape) / bg_class_ind
#     soft_label[pos, :] = 0
#     soft_label[pos, pos_label] = 1
#     pred_softmax = pred.softmax(-1)
#     scale_factor = soft_label - pred_softmax
#     log_likelihood = - F.log_softmax(pred, dim = 1) * soft_label * scale_factor.abs().pow(beta)
#     loss = log_likelihood.sum(-1)

#     if weight is not None:
#         weight = weight.float()
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss

# def calculate_loss_func(pred,
#                        target,
#                        weight=None,
#                        reduction='mean',
#                        avg_factor=None):
    
#     bg_class_ind = pred.shape[-1] # bg
    
#     soft_label = pred.new_ones(pred.shape) / bg_class_ind
#     relation_loss = - F.log_softmax(pred, dim = 1) * soft_label
#     individual_loss = F.binary_cross_entropy_with_logits(
#         pred, torch.zeros_like(pred, dtype=torch.float, device=pred.device), reduction='none')
#     loss = torch.sum(relation_loss + individual_loss, dim=-1)

#     pos = ((target >= 0) & (target < bg_class_ind)).nonzero().squeeze(1)
#     loss[pos] = F.cross_entropy(
#         pred[pos, :],
#         target[pos],
#         reduction='none',)
    
#     loss = loss / bg_class_ind

#     if weight is not None:
#         weight = weight.float()
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss

# outputs = torch.zeros_like(class_logits)
# den = torch.logsumexp(class_logits, dim=1)  # B, H, W       den of softmax
# outputs[:, 0] = torch.logsumexp(class_logits[:, 0:self.n_old_cl+1], dim=1) - den  # B, H, W       p(O)
# outputs[:, self.n_old_cl+1:] = class_logits[:, self.n_old_cl+1:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

# labels = labels.clone()  # B, H, W

# classification_loss = F.nll_loss(outputs, labels)


# def calculate_loss_func(num_classes, pred,
#                        target,
#                        weight=None,
#                        reduction='mean',
#                        avg_factor=None):
    
#     all_class = pred.shape[-1] # bg
    
#     soft_label = pred.new_ones(pred.shape) / all_class
#     relation_loss = - F.log_softmax(pred, dim = 1) * soft_label
#     loss = torch.mean(relation_loss, dim=-1)

#     pos = ((target >= 0) & (target < num_classes)).nonzero().squeeze(1)
#     loss[pos] = F.cross_entropy(
#         pred[pos, :],
#         target[pos],
#         reduction='none',)

#     if weight is not None:
#         weight = weight.float()
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss

# def _expand_onehot_labels(labels, label_channels):
#     """Expand onehot labels to match the size of prediction."""
#     bin_labels = labels.new_full((labels.size(0), label_channels), 0)
#     valid_mask = labels >= 0
#     inds = torch.nonzero(
#         valid_mask & (labels < label_channels), as_tuple=False)

#     if inds.numel() > 0:
#         bin_labels[inds, labels[inds]] = 1

#     return bin_labels

# def calculate_loss_func(num_classes, pred,
#                        target,
#                        weight=None,
#                        reduction='mean',
#                        avg_factor=None):

#     outputs = torch.zeros((pred.shape[0], num_classes + 1), device=pred.device, dtype=pred.dtype)
#     den = torch.logsumexp(pred, dim=1)  # den of softmax
#     outputs[:, num_classes] = torch.logsumexp(pred[:, num_classes:], dim=1) - den  # other class
#     outputs[:, :num_classes] = pred[:, :num_classes] - den.unsqueeze(dim=1)  # base class

#     loss = F.nll_loss(outputs, target, reduction='none')
#     # loss = classification_loss.sum(-1)
#     pos = ((target >= 0) & (target < num_classes)).nonzero().squeeze(1)
#     loss[pos] = F.cross_entropy(
#         pred[pos, :],
#         target[pos],
#         reduction='none',)

#     if weight is not None:
#         weight = weight.float()
#     loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
#     return loss

def calculate_loss_func(num_classes, pred,
                       target,
                       weight=None,
                       bg_inds=None,
                       reduction='mean',
                       avg_factor=None):
    # ori_score = pred[:, :num_classes + 1]
    # pred[bg_flags.bool(), num_classes+1:] = float('-inf')
    if pred.shape[-1] > num_classes + 1:
        pred = pred - torch.max(pred.clone().detach(), dim=-1, keepdim=True)[0] # for numerical stability
        outputs = torch.zeros((pred.shape[0], num_classes + 1), device=pred.device, dtype=pred.dtype)
        den = torch.logsumexp(pred, dim=1)  # den of softmax
        outputs[:, num_classes] = torch.logsumexp(pred[:, num_classes:], dim=1) - den  # other class
        outputs[:, :num_classes] = pred[:, :num_classes] - den.unsqueeze(dim=1)  # base class

        loss = F.nll_loss(outputs, target, reduction='none')
    else:
        loss = F.cross_entropy(
            pred,
            target,
            reduction='none',)
    if bg_inds is not None:
        loss[bg_inds] = F.cross_entropy(
            pred[bg_inds],
            target[bg_inds],
            reduction='none',)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class CloseWorldLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 beta=2.0,):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.\
        """
        super(CloseWorldLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                num_classes,
                pred,
                target,
                weight=None,
                bg_inds=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * calculate_loss_func(
            num_classes,
            pred,
            target,
            weight,
            bg_inds,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls
