_base_ = [
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\models\upernet_r50.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\my_projects\FH-PS-AOP\configs\dataset.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\schedules\schedule_80k.py',
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
        loss_decode=dict(
            type='DiceLoss', loss_weight=3.0, pixel_weight='neighbor_s')
        ),
    auxiliary_head=dict(
        num_classes=3),
)
train_dataloader = dict(
    batch_size=4,
    num_workers=4
)
