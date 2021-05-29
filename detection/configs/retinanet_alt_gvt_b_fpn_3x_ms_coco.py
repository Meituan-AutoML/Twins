_base_ = './retinanet_pcpvt_s_fpn_3x_ms_coco.py'

model = dict(
    pretrained='pretrained/alt_gvt_base.pth',
    backbone=dict(
        type='alt_gvt_base',
        style='pytorch'),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        out_channels=256,))

