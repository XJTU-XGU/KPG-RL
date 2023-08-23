# Keypoint-Guided Optimal Transport (NeurIPS)
Official codes for conference paper ["Keypoint-Guided Optimal Transport with Applications in Heterogeneous Domain Adaptation"](https://openreview.net/forum?id=m6DJxSuKuqF&noteId=SEp6zzXmpLE) and journal version ["Keypoint-Guided Optimal Transport"](https://arxiv.org/abs/2303.13102).
![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure.png)

We presented a new optimal transport model named KPG-RL that leverages a few prior annotated keypoints to guide the matching of other points in OT. We propose a mask-based modeling of the transport plan and to preserve the relation of each data point to keypoints to realize the guidance. 

Mask-based transport plan:

![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure2.png)

Relation modeling:

![](https://github.com/XJTU-XGU/KPG-RL/blob/main/figures/figure3.png)

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
__UDA__: codes for unsupervised DA experiments (Appendix B.3) <br>
__toy_data__: codes for toy experiments for KPG-RL (Section 5.1) <br>
__toy_data_partial__: codes for toy experiments for Partial KPG-RL (Appendix B.1) <br>

Please follow the __Readme.md__ in each folder to run the codes. 

## News
[2023.04] __We provide a easy-to-follow demo in [demo.ipynb](https://github.com/XJTU-XGU/KPG-RL/blob/main/demo.ipynb), which includes the model, algorithm, and codes. Welcome to try it!__

[2023.06] __We upload the keypoint-guided OT packadge in folder ["keypoint_guided_optimal_transport"](https://github.com/XJTU-XGU/KPG-RL/tree/main/keypoint_guided_optimal_transport) and examples in ["Examples_of_KPG_OT.py"](https://github.com/XJTU-XGU/KPG-RL/blob/main/Examples_of_KPG_OT.py).__

[2023.08] __The linear programming implementation on sparse matrixes is available in ["keypoint_guided_optimal_transport/linearprog.py"](https://github.com/XJTU-XGU/KPG-RL/tree/main/keypoint_guided_optimal_transport/linearprog.py), which is memory-efficient for a large number of data points. Please set "sparse=True"  in function "lp" to use it as follows:__
```python
lp(p, q, C, Mask=None,sparse=True)
```

## Using keypoint-guided OT in you code
For KPG-RL model, use the following code
``` python
# import keypoint-guided OT
from keypoint_guided_optimal_transport.keypoint_guided_OT import KeyPointGuidedOT

# define the samples xs and xt , the mass p and q, and the keypoint index pair K
xs =  # ndarray with shape (m,d)
xt =  # ndarray with shape (n,d)
p =   # ndarray with shape (m,)
q =   # ndarray with shape (n,)
K =   # list of tuples, e.g., [(0,0),(10,20)]

kgot = KeyPointGuidedOT()
pi = kgot.kpg_rl(p,q,xs,xt,K,cost_function="L2",algorithm="linear_programming",tau_s=0.1,tau_t=0.1,normalized=True,
               reg=0.0001,max_iterations=100000,thres=1e-5,eps=1e-10)
# pi is a ndarray with shape (m,n). The algorithm could be "linear_programming" or "sinkhorn".
```

For KPG-RL-KP model, use the following code
``` python
# import keypoint-guided OT
from keypoint_guided_optimal_transport.keypoint_guided_OT import KeyPointGuidedOT

# define the samples xs and xt , the mass p and q, and the keypoint index pair K
xs =  # ndarray with shape (m,d)
xt =  # ndarray with shape (n,d)
p =   # ndarray with shape (m,)
q =   # ndarray with shape (n,)
K =   # list of tuples, e.g., [(0,0),(10,20)]
alpha =   # combination coeffecient

kgot = KeyPointGuidedOT()
pi = kgot.kpg_rl_kp(p,q,xs,xt,K,alpha=alpha,cost_function="L2",algorithm="linear_programming",tau_s=0.1,tau_t=0.1,
                  normalized=True,reg=0.0001,max_iterations=100000,eps=1e-10,thres=1e-5)
# pi is a ndarray with shape (m,n). The algorithm could be "linear_programming" or "sinkhorn".
```

For KPG-RL-GW model, use the following code
``` python
# import keypoint-guided OT
from keypoint_guided_optimal_transport.keypoint_guided_OT import KeyPointGuidedOT

# define the samples xs and xt , the mass p and q, and the keypoint index pair K
xs =  # ndarray with shape (m,d)
xt =  # ndarray with shape (n,d)
p =   # ndarray with shape (m,)
q =   # ndarray with shape (n,)
K =   # list of tuples, e.g., [(0,0),(10,20)]
alpha =   # combination coeffecient

kgot = KeyPointGuidedOT()
pi = kgot.kpg_rl_gw(p,q,xs,xt,K,alpha=alpha,cost_function="L2",algorithm="linear_programming",tau_s=0.1,tau_t=0.1,
                  normalized=True,reg=0.0001,max_iterations=100000,eps=1e-10,thres=1e-5)
# pi is a ndarray with shape (m,n). The algorithm could be "linear_programming" or "sinkhorn".
```

For partial-KPG-RL model, use the following code
``` python
# import keypoint-guided OT
from keypoint_guided_optimal_transport.keypoint_guided_OT import KeyPointGuidedOT

# define the samples xs and xt , the mass p and q, and the keypoint index pair K
xs =  # ndarray with shape (m,d)
xt =  # ndarray with shape (n,d)
p =   # ndarray with shape (m,)
q =   # ndarray with shape (n,)
K =   # list of tuples, e.g., [(0,0),(10,20)]
s =   # total mass 

kgot = KeyPointGuidedOT()
pi = kgot.partial_kpg_rl(p, q, xs, xt, K, s=0.5, cost_function="L2", tau_s=1.0,
                  tau_t=1.0, normalized=False,eps=1e-10)
```

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

@misc{gu2023keypointguided,
      title={Keypoint-Guided Optimal Transport}, 
      author={Xiang Gu and Yucheng Yang and Wei Zeng and Jian Sun and Zongben Xu},
      year={2023},
      eprint={2303.13102},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
For any problem, please do not hesitate to contact xianggu@stu.xjtu.edu.cn.


The code is based on code of [POT packadge](https://pythonot.github.io/).
