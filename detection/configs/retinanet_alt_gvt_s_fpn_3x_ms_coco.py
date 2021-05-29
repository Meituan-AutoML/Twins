_base_ = './retinanet_pcpvt_s_fpn_3x_ms_coco.py'

model = dict(
    pretrained='pretrained/alt_gvt_small.pth',
    backbone=dict(
        type='alt_gvt_small',
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256)
)
