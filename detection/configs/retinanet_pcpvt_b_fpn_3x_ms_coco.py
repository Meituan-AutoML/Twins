_base_ = './retinanet_pcpvt_s_fpn_3x_ms_coco.py'

model = dict(
    pretrained='pretrained/pcpvt_base.pth',
    backbone=dict(
        type='pcpvt_base',
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 320, 512],
        out_channels=256,))
