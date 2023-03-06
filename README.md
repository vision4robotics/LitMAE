# LitMAE: Masked Auto-Emphasizer for Nighttime Aerial Tracking　

Code and demo videos of LitMAE --- Lighten up the semantics for aerial trackers.

# Abstract 
>The development of nighttime tracking has sig- nificantly advanced the expansive all-day applications of in- telligent unmanned aerial vehicles (UAVs). However, previous state-of-the-art (SOTA) UAV trackers have lacked emphasis on latent semantics, resulting in a disability to cope with complex nighttime conditions, e.g., frequent viewpoint change, fast motion, and sudden illumination variation. To address this, we propose a novel framework, i.e., LitMAE, to directly auto- emphasize the latent semantics and adaptively lighten up aerial images for nighttime UAV tracking. By fusing distinguishable semantic information, LitMAE generates different enhance- ment strategies for background and foreground. Specifically, a novel pixel-level masked image modeling (PixMIM) process is proposed to elevate the enhancer’s capability for semantics perception. We also construct a static-dynamic two-branch module to discriminate the background and foreground with latent semantics auto-emphasis, and a dynamic-weight block based on the Transformer for efficient foreground perception. Extensive evaluations on authoritative benchmarks demonstrate that LitMAE supports SOTA UAV trackers to achieve com- petitive accuracy and robustness in nighttime scenes, outper- forming previous SOTA low-light enhancers. Real-world tests further validate the practicability and efficiency of the proposed method.
![The proposed framework](https://github.com/vision4robotics/LitMAE/blob/main/images/Framework.jpg)

# Demo video

[![LitMAE](https://github.com/vision4robotics/LitMAE/blob/main/images/demo.png)](https://github.com/vision4robotics/LitMAE/blob/main/images/demo.png)

<!-- # Publication and citation

LitMAE is proposed in our paper accepted by IROS 2021. Detailed explanation of our method can be found in the paper:

Junjie Ye, Changhong Fu, Guangze Zheng, Ziang Cao, and Bowen Li

**DarkLighter: Light up the Darkness for UAV Tracking**

In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021

Please cite the above publication if you find this work helpful. Bibtex entry:

> @Inproceedings{Ye2021IROS,
>
> title={{DarkLighter: Light up the Darkness for UAV Tracking}},
>
> author={Ye, Junjie and Fu, Changhong and Zheng, Guangze and Cao, Ziang and Li, Bowen},  
>
> booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
>
> year={2021}, 
>
> pages={1-7}}
 -->
# Contact 
Weiyu Peng

Email: 2032698@tongji.edu.cn

Changhong Fu

Email: changhongfu@tongji.edu.cn

# Demonstration running instructions

### Requirements

1.Python 3.7

2.Pytorch 1.0.0

3.opencv-python

4.torchvision

5.cuda 10.2

6.timm

>Download the package and extract it.
>
>1. Train and test the proposed method in [LitMAE](https://github.com/vision4robotics/LitMAE/tree/main/LitMAE).
>
>2. Track with LitMAE in [TrackingPipeline](https://github.com/vision4robotics/LitMAE/tree/main/TrackingPipeline).



# Acknowledgements

We sincerely thank the contribution of `Junjie Ye et al.`, `Kaiming He et al.` for their previous work [DarkLighter](https://github.com/vision4robotics/DarkLighter) and [MAE](https://github.com/facebookresearch/mae).

