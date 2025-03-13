# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import torch.nn.functional as F
import math


class ModelEMA:
    def __init__(self, category_embeddings: torch.Tensor, classid_to_idx, decay=0.9999, updates=0):
        # Create EMA
        self.category_embeddings = category_embeddings.clone()
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.classid_to_idx = classid_to_idx

    def update(self, labels, embeddings):
        # Update EMA parameters
        embeddings = embeddings.clone().detach()
        with torch.no_grad():
            self.updates += 1
            inds = self.classid_to_idx[labels]
            d = self.decay(self.updates)
            self.category_embeddings[inds] = self.category_embeddings[inds] * d + embeddings * (1-d)



@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 num_use_pseudo_box_epoch,
                 add_pseudo_box_to_rpn,
                 embedding_file,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_use_pseudo_box_epoch = num_use_pseudo_box_epoch
        self.add_pseudo_box_to_rpn = add_pseudo_box_to_rpn
        self.epoch = -1
        self.use_pseudo_box = self.roi_head.use_pseudo_box
        self.use_text_space_rkd_loss = self.roi_head.use_text_space_rkd_loss

        # only for ov-coco dataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        novel_inds = list(range(48, 65))
        base_inds = list(range(0, 48))
        self.base_inds_tensor = torch.tensor(base_inds, device=device)
        self.novel_inds_tensor = torch.tensor(novel_inds, device=device)
        classid_to_inds = torch.zeros((65,), dtype=torch.long, device=device)
        classid_to_inds[self.base_inds_tensor] = torch.arange(0, len(self.base_inds_tensor), device=device)
        category_embeddings = torch.load(embedding_file, 'cpu').float().to(device)
        self.category_embeddings = F.normalize(category_embeddings, dim=1)
        if self.use_text_space_rkd_loss:
            self.roi_head.class_centroid = ModelEMA(category_embeddings[self.base_inds_tensor], classid_to_inds)
            self.roi_head.category_embeddings = self.category_embeddings


    def set_epoch(self, epoch): 
        if epoch + 1 < self.num_use_pseudo_box_epoch:
            self.roi_head.use_pseudo_box = False
            self.roi_head.use_text_space_rkd_loss = False
        if epoch > self.epoch and epoch + 1 >= self.num_use_pseudo_box_epoch and self.use_pseudo_box and (not self.roi_head.use_pseudo_box):
            print("Begin to use pseudo box !!!!")
            # print("Close image level distill !!!!")
            self.roi_head.use_pseudo_box = True
            self.roi_head.use_text_space_rkd_loss = self.use_text_space_rkd_loss
            # self.roi_head.use_image_level_distill = False
        self.epoch = epoch
        print(self.epoch)


    def forward_train(self,
                      img,
                      img_no_normalize,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        idxs = []
        for j, img_meta in enumerate(img_metas):
            idxs.append(self.base_inds_tensor.clone().to(gt_labels[j].device))

        num_base_classes = len(idxs[0])
        num_imgs = len(img_metas)
        class_mapping = torch.ones((num_imgs, self.category_embeddings.shape[0]), dtype=gt_labels[0].dtype, device=gt_labels[0].device) * -1
        remapping_gt_labels = []
        for i in range(num_imgs):
            class_mapping[i, idxs[i]] = torch.arange(0, num_base_classes, dtype=gt_labels[i].dtype, device=gt_labels[i].device)
            remapping_gt_labels.append(class_mapping[i, gt_labels[i]])
        
        class_prompt = []
        for j, img_meta in enumerate(img_metas):
            class_prompt.append(self.category_embeddings[idxs[j]])
            # class_prompt.append(self.category_embeddings[idxs[j]])
        class_prompt = torch.stack(class_prompt)

        x = self.extract_feat(img)

        losses = dict()
        rpn_gt_bboxes = []
        for i in range(num_imgs):
            if self.add_pseudo_box_to_rpn:
                size_mask = ((proposals[i][:, 2] - proposals[i][:, 0]) > 32) & ((proposals[i][:, 3] - proposals[i][:, 1]) > 32)
                valid_proposal = proposals[i][size_mask, :4]
                rpn_gt_bboxes.append(torch.cat([gt_bboxes[i], valid_proposal], dim=0))
            else:
                rpn_gt_bboxes.append(gt_bboxes[i])

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                rpn_gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 img_no_normalize,
                                                 proposals,
                                                 remapping_gt_labels,
                                                 class_prompt,
                                                 idxs,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
    
    def simple_test(self, img, img_no_normalize, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        
        idxs = []
        for j, img_meta in enumerate(img_metas):
            idxs.append(torch.arange(0, self.category_embeddings.shape[0], dtype=torch.long, device=img.device))
        
        class_prompt = []
        for j, img_meta in enumerate(img_metas):
            class_prompt.append(self.category_embeddings[idxs[j]])
            # class_prompt.append(self.category_embeddings[idxs[j]])
        class_prompt = torch.stack(class_prompt)

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, 
            img_no_normalize,
            class_prompt,
            self.base_inds_tensor,
            self.novel_inds_tensor,
            rescale=rescale)