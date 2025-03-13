_base_ = './myadamixer_r50_1x_coco.py'
pretrained = '../ovd_resources/swin_base_patch4_window7_224_22k.pth'
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(in_channels=[128, 256, 512, 1024])
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.000025,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            # Swin-related settings
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

lr_config = dict(warmup_iters=1000)
