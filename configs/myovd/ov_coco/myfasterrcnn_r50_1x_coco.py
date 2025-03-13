_base_ = [
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]


data_root = '../dataset/coco/'
work_dir_prefix = 'work_dirs/myadamixer_mmdet'
log_interval = 100

IMAGE_SCALE = (1333, 800)

# dataset settings
dataset_type = 'OVCocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=IMAGE_SCALE, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_no_normalize', 'proposals', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMAGE_SCALE,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_no_normalize']),
            dict(type='Collect', keys=['img', 'img_no_normalize']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.48.json',
        img_prefix=data_root + 'train2017/',
        proposal_file='../ovd_resources/coco_proposal_train_object_centric.pkl',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.65.min.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.65.min.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
)


TEXT_DIM = 512

model = dict(
    type='FasterRCNN',
    num_use_pseudo_box_epoch=-1,
    embedding_file='../ovd_resources/coco_detpro_category_embeddings_vit-b-32.pt',
    add_pseudo_box_to_rpn=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='../ovd_resources/torchvision_resnet50.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(
        type='StandardRoIHead',
        max_pseudo_box_num=5,
        cls_tau=50,
        skd_tau=20,
        rkd_tau=5,
        pre_extracted_clip_text_feat='../ovd_resources/coco_proposals_text_embedding10/',
        use_pseudo_box=True,
        split_visual_text=True,
        use_text_space_rkd_loss=False,
        loss_visual_skd=dict(type='CrossEntropyLoss', loss_weight=0.1),
        loss_visual_rkd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1.0),
        use_image_level_distill=True,
        novel_obj_queue_dict=dict(
            names=['obj_query', 'clip_query', 'backbone_image_query', 'clip_image_query'],
            lengths=[2048, 2048, 512, 512], 
            emb_dim=TEXT_DIM),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),

    # training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')]

evaluation = dict(metric=['bbox'], interval=1)

log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

def __date():
    import datetime
    return datetime.datetime.now().strftime('%m%d_%H%M')

postfix = '_' + __date()

find_unused_parameters = True

load_from = None #'../ovd_resources/epoch_2.pth'
resume_from = None #'work_dirs/myadamixer_r50_1x_coco/epoch_6.pth'


