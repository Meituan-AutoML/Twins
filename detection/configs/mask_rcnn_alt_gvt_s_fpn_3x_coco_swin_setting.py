_base_ = './mask_rcnn_pcpvt_s_fpn_3x_coco_swin_setting.py'

model = dict(
    pretrained='pretrained/alt_gvt_small.pth',
    backbone=dict(
        type='alt_gvt_small',
        style='pytorch'),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        out_channels=256,))
