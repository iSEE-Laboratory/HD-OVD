# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
from typing import Any
import numpy as np
from collections import OrderedDict
from collections.abc import Sequence
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

from .api_wrappers import COCOeval
from .builder import DATASETS
from .coco import CocoDataset


def _recalls(all_ious, proposal_nums, thrs):

    img_num = len(all_ious)
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((len(proposal_nums), total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            if len(all_ious[i]) == 0:
                continue
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.shape[1] != 0:
                for j in range(ious.shape[0]):
                    gt_max_overlaps = ious.argmax(axis=1)
                    max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                    gt_idx = max_ious.argmax()
                    gt_ious[gt_idx] = max_ious[gt_idx]
                    box_idx = gt_max_overlaps[gt_idx]
                    ious[gt_idx, :] = -1
                    ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    # _ious = np.fliplr(np.sort(_ious, axis=1))
    matched = np.zeros((len(thrs), len(proposal_nums), total_gt_num))
    for i, thr in enumerate(thrs):
        matched[i] = _ious >= thr

    return matched




def eval_recalls(gts,
                 proposals,
                 proposal_nums=[100],
                 iou_thrs=[0.5],
                 logger=None,
                 use_legacy_coordinate=False):
    """Calculate recalls.

    Args:
        gts (list[ndarray]): a list of arrays of shape (n, 4)
        proposals (list[ndarray]): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums (int | Sequence[int]): Top N proposals to be evaluated.
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        use_legacy_coordinate (bool): Whether use coordinate system
            in mmdet v1.x. "1" was added to both height and width
            which means w, h should be
            computed as 'x2 - x1 + 1` and 'y2 - y1 + 1'. Default: False.


    Returns:
        ndarray: recalls of different ious and proposal nums
    """

    img_num = len(gts)
    assert img_num == len(proposals)
    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(
                gts[i],
                img_proposal[:prop_num, :4],
                use_legacy_coordinate=use_legacy_coordinate)
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)
    return recalls



@DATASETS.register_module()
class OVCocoDataset(CocoDataset):

    bases=(
        'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
        'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
        'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
        'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
        'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase',
        'toothbrush'
    )
    novels=(
        'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
        'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
        'sink', 'scissors'
    )

    CLASSES = bases + novels

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        gt_labels = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                gt_labels.append(np.zeros((0)))
                continue
            bboxes = []
            labels = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
                labels.append(self.cat2label[ann['category_id']])
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
                labels = np.zeros((0))
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)

        gt_labels = np.hstack(gt_labels)
        #print(gt_labels.shape)
        novel_gt_labels = gt_labels >= len(self.bases)
        #print(novel_gt_labels.shape)
        base_gt_labels = gt_labels < len(self.bases)
        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = np.sum(recalls, axis=-1) / len(gt_labels)
        ar = ar.mean(axis=1)
        #print(recalls.shape)
        ar_novel = np.sum(recalls[:, :, novel_gt_labels], axis=-1) / np.sum(novel_gt_labels)
        ar_novel = ar_novel.mean(axis=1)

        ar_base = np.sum(recalls[:, :, base_gt_labels], axis=-1) / np.sum(base_gt_labels)
        ar_base = ar_base.mean(axis=1)
        return ar, ar_base, ar_novel

    def summarize(self, cocoEval: COCOeval, prefix: str):
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io):
            cocoEval.summarize()
        print(f'Evaluate *{prefix}*\n{string_io.getvalue()}')

        stats = {
            s: f'{cocoEval.stats[i]:.04f}'
            for i, s in enumerate(['', '50', '75', 's', 'm', 'l'])
        }
        stats['copypaste'] = ' '.join(stats.values())
        return {f'{prefix}_bbox_mAP_{k}': v for k, v in stats.items()}

    def evaluate(self, results, metric='bbox', *args, **kwargs):
        
        if metric[0] == 'bbox':
            results = self._det2json(results)
            try:
                results = self.coco.loadRes(results)
            except IndexError:
                print('The testing results is empty')
                return dict()
        
            coco_eval = COCOeval(self.coco, results, 'bbox')
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = [100, 300, 1000]

            coco_eval.evaluate()
            coco_eval.accumulate()

            # iou_thrs x recall x k x area x max_dets
            precision: np.ndarray = coco_eval.eval['precision']
            # iou_thrs x k x area x max_dets
            recall: np.ndarray = coco_eval.eval['recall']
            assert len(self.cat_ids) == precision.shape[2] == recall.shape[1], (
                f"{len(self.cat_ids)}, {precision.shape}, {recall.shape}"
            )

            all_ = self.summarize(
                coco_eval, f'COCO_{len(self.bases)}_{len(self.novels)}'
            )

            coco_eval.eval['precision'] = precision[:, :, :len(self.bases), :, :]
            coco_eval.eval['recall'] = recall[:, :len(self.bases), :, :]
            bases = self.summarize(coco_eval, f'COCO_{len(self.bases)}')

            coco_eval.eval['precision'] = precision[:, :, len(self.bases):, :, :]
            coco_eval.eval['recall'] = recall[:, len(self.bases):, :, :]
            novels = self.summarize(coco_eval, f'COCO_{len(self.novels)}')

            all_.update(bases)
            all_.update(novels)

            return all_
        elif metric[0] == 'recall':
            #print(len(results))
            #print(len(results[0]))
            proposal_nums=(100,)
            iou_thrs = np.array([0.5])
            eval_results = OrderedDict()
            if isinstance(results[0], tuple):
                raise KeyError('proposal_fast is not supported for '
                                'instance segmentation result.')
            results_fuse = [np.vstack(res) for res in results]
            #print(results_fuse[0].shape)
            ar, ar_base, ar_novel = self.fast_eval_recall(
                results_fuse, proposal_nums, iou_thrs, logger='silent')
            for i, num in enumerate(proposal_nums):
                eval_results[f'AR@{num}'] = ar[i]
                print(f'\nAR@{num}\t{ar[i]:.4f}')
            for i, num in enumerate(proposal_nums):
                eval_results[f'ARb@{num}'] = ar_base[i]
                print(f'\nARb@{num}\t{ar_base[i]:.4f}')
            for i, num in enumerate(proposal_nums):
                eval_results[f'ARn@{num}'] = ar_novel[i]
                print(f'\nARn@{num}\t{ar_novel[i]:.4f}')
            return eval_results
        else:
            raise NotImplementedError
