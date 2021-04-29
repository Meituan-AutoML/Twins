# Twins: Revisiting the Design of Spatial Attention in Vision Transformers


Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins- PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including image- level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks.

![Twins-SVT-S](twins_svt_s.png)
Figure 1. Twins-SVT-S Architecture (Right side shows the inside of two consecutive Transformer Encoders).

## Model Zoo

### Image Classification

We provide baseline Twins models pretrained on ImageNet 2012.

| Name | Alias in paper | acc@1 | FLOPs(G)|#params (M) | url |
| --- | --- | --- | --- | --- |--- |
| PVT+CPVT-Small| Twins-PCPVT-S | 81.2 | 3.7 | 24.1 | [pcpvt_small.pth](https://drive.google.com/file/d/1TWIx_8M-4y6UOKtbCgm1v-UVQ-_lYe6X/view?usp=sharing)
| PVT+CPVT-Base| Twins-PCPVT-B | 82.7 | 6.4 | 43.8 | [pcpvt_base.pth](https://drive.google.com/file/d/1BsD3ZRivvPsHoZB1AX-tbirFLtCln8ky/view?usp=sharing)
| ALT-GVT-Small | Twins-SVT-S | 81.3 | 2.8| 24 | [alt_gvt_small.pth](https://drive.google.com/file/d/131SVOphM_-SaBytf4kWjo3ony5hpOt4S/view?usp=sharing)|
| ALT-GVT-Base | Twins-SVT-B| 83.1 | 8.3 | 56 | [alt_gvt_base.pth](https://drive.google.com/file/d/1s83To8xgDWY6Ad8VBP3Nx9gqY709rrGu/view?usp=sharing)|
| ALT-GVT-Large | Twins-SVT-L | 83.3 | 14.8 | 99.2 |[alt_gvt_large.pth](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view?usp=sharing)|



^ Note: Our code will be released soon.


### Citation

```
@article{chu2021Twins,
	title={Twins: Revisiting the Design of Spatial Attention in Vision Transformers},
	author={Xiangxiang Chu and Zhi Tian and Yuqing Wang and Bo Zhang and Haibing Ren and Xiaolin Wei and Huaxia Xia and Chunhua Shen},
	journal={Arxiv preprint 2104.13840},
	url={https://arxiv.org/pdf/2104.13840.pdf},
	year={2021}
}
```



