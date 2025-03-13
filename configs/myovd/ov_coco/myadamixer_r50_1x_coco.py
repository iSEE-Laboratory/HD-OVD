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

num_stages = 6
num_query = 300
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048

# P_in for spatial mixing in the paper.
in_points_list = [32, ] * num_stages

# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns_list = [128, ] * num_stages

# G for the mixer grouping in the paper. Please distinguishe it from num_heads in MHSA in this codebase.
n_group_list = [4, ] * num_stages

TEXT_DIM = 512

model = dict(
    type='QueryBased',
    pretrained='../ovd_resources/torchvision_resnet50.pth',
    dataset='ov_coco',
    rpn_loss=True,
    num_use_pseudo_box_epoch=2,
    embedding_file='../ovd_resources/coco_detpro_category_embeddings_vit-b-32.pt',
    base_ind_file=None,
    add_pseudo_box_to_rpn=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    # backbone=dict(
    #     type='CLIPCNNBackbone',
    #     pretrained_weight='../ovd_resources/CLIP_RN50.pt',
    #     use_text_space_feature=False),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='PromptAnchorHead2',
        # text_dim=TEXT_DIM,
        clip_obj_embedding_file='../ovd_resources/obj_detpro_category_embeddings_vit-b-32.pt',
        in_channels=256,
        strides=[8, 16, 32, 64],
        feat_channels=256,
        num_query=num_query,),
    roi_head=dict(
        type='AdaMixerDecoderPrompt',
        featmap_strides=[4, 8, 16, 32],
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=QUERY_DIM,
        text_dim=TEXT_DIM,
        clip_model_path='../ovd_resources/CLIP_ViT-B-32.pt',
        max_pseudo_box_num=5,
        cls_tau=50,
        skd_tau=20,
        rkd_tau=5,
        pre_extracted_clip_text_feat='../ovd_resources/coco_proposals_text_embedding10/',
        use_pseudo_box=True,
        split_visual_text=True,
        use_text_space_rkd_loss=True,
        loss_visual_skd=dict(type='CrossEntropyLoss', loss_weight=0.5),
        loss_visual_rkd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=5.0),
        use_image_level_distill=True,
        num_additional_padding_prompts=32,
        novel_obj_queue_dict=dict(
            names=['novel_obj', 'obj_query_0', 'obj_query_1', 'obj_query_2', 'obj_query_3', 'obj_query_4', 'obj_query_5', 'clip_query_0', 'clip_query_1', 'clip_query_2', 'clip_query_3', 'clip_query_4', 'clip_query_5', 'backbone_image_query', 'clip_image_query'],
            lengths=[2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 512, 512], 
            emb_dim=TEXT_DIM),
        #base_ind_file=None,
        bbox_head=[
            dict(
                type='AdaMixerDecoderStagePrompt',
                num_classes=65,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=FF_DIM,
                content_dim=QUERY_DIM,
                feat_channels=FEAT_DIM,
                dropout=0.0,
                in_points=in_points_list[stage_idx],
                out_points=out_patterns_list[stage_idx],
                n_groups=n_group_list[stage_idx],
                text_dim=TEXT_DIM,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(type='CloseWorldLoss', loss_weight=0.5),
                loss_obj=dict(type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0,),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for stage_idx in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1),
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=dict(), rcnn=dict(max_per_img=100, mask_thr_binary=0.5)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00005,
    weight_decay=0.0001,
)
#optimizer = dict(
#    _delete_=True,
#    type='AdamW',
#    lr=0.0001,
#    weight_decay=0.0001,
#    paramwise_cfg=dict(
#    custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2),
)

#optim_wrapper = dict(
#    type='OptimWrapper',
#    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
#    clip_grad=dict(max_norm=0.1, norm_type=2),
#    paramwise_cfg=dict(
#        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

# learning policy
lr_config = dict(
    policy='step',
    step=[8,11],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)

# custom hooks
custom_hooks = [dict(type='SetEpochInfoHook')]

runner = dict(type='EpochBasedRunner', max_epochs=12)
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


