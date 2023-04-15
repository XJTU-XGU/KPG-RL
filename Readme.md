# Keypoint-Guided Optimal Transport
Official codes for ["Keypoint-Guided Optimal Transport with Applications in Heterogeneous Domain Adaptation"](https://openreview.net/forum?id=m6DJxSuKuqF&noteId=SEp6zzXmpLE).
![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure.png)

We presented a new optimal transport model named KPG-RL that leverages a few prior annotated keypoints to guide the matching of other points in OT. We propose a mask-based modeling of the transport plan and to preserve the relation of each data point to keypoints to realize the guidance. 

Mask-based transport plan:

![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure2.png)

Relation modeling:

![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure3.png = 683*784)

With the keypoints, our approach apparently improves the matching accuracy.
![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure4.png)

## Requirements
python3.6 <br>
scipy==1.7.1 <br>
numpy==1.20.3 <br>
matplotlib==3.4.3 <br>
cvxpy==1.2.0 <br>
pot==0.8.1.0 <br>
pytorch==1.5.0 <br>

## Instructions for the folders
__HDA__: codes for HDA experiments (Section 5.2) <br>
__Open-Set HDA__: codes for Open-Set HDA experiments (Section 5.3) <br>
__UDA__: codes for Open-Set HDA experiments (Section 5.3) <br>
__toy_data__: codes for toy experiments for KPG-RL (Section 5.1) <br>
__toy_data_partial__: codes for toy experiments for Partial KPG-RL (Appendix B.1) <br>

Please follow the __Readme.md__ in each folder to run the codes. 

## Citation:
```
@inproceedings{
gu2022keypointguided,
title={Keypoint-Guided Optimal Transport with Applications in Heterogeneous Domain Adaptation},
author={Xiang Gu and Yucheng Yang and Wei Zeng and Jian Sun and Zongben Xu},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=m6DJxSuKuqF}
}
```

## Contact
For any problem, please do not hesitate to contact xianggu@stu.xjtu.edu.cn.


The code is based on code of [POT packadge](https://pythonot.github.io/).
