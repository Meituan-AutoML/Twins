_base_ = './retinanet_pcpvt_s_fpn_3x_ms_coco.py'

model = dict(
    pretrained='pretrained/alt_gvt_large.pth',
    backbone=dict(
        type='alt_gvt_large',
        style='pytorch'),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        out_channels=256,))




