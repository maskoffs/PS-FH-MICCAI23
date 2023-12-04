_base_ = [
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\models\upernet_r50.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\my_projects\FH-PS-AOP\configs\dataset.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\schedules\schedule_40k.py',
    r'C:\Users\qiuyaoyang\PycharmProjects\mmsegmentation\configs\_base_\default_runtime.py'
]
crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[44.235, 44.235, 44.235],
    std=[44.688, 44.688, 44.688],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=3,
        ),
    auxiliary_head=dict(
        num_classes=3),
)