_base_ = [
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\models\upernet_r50.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\my_projects\FH-PS-AOP\configs\dataset_sequent.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\schedules\schedule_40k.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\default_runtime.py'
]
crop_size = (256, 256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    # image > 0
    mean=[44.235, 44.235, 44.235],
    std=[44.688, 44.688, 44.688],
    # # image
    # mean=[26.130, 26.130, 26.130],
    # std=[40.654, 40.654, 40.654],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(type='TverskyLoss', loss_weight=3.0, ignore_index=0, alpha=0.4, beta=0.6),
            dict(type='CrossEntropyLoss', loss_weight=1.)]
        ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=0.4),
    ),
)
train_dataloader = dict(
    batch_size=4,
    num_workers=4
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=40000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005),
#     clip_grad=None)
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=0.0001,
#         power=0.9,
#         begin=0,
#         end=40000,
#         by_epoch=False)
# ]

# load_from = 'work_dirs/upernet_r101_4xb4-dice-3.0-80k_psfh-256x256/iter_80000.pth'