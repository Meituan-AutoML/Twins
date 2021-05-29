_base_ = './mask_rcnn_pcpvt_s_fpn_1x_coco_pvt_setting.py'

model = dict(
    pretrained='pretrained/alt_gvt_large.pth',
    backbone=dict(
        type='alt_gvt_large',
        style='pytorch'),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        out_channels=256,))

