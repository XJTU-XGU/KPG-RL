#Keypoint-Guided Optimal Transport
Official codes for ["Keypoint-Guided Optimal Transport with Applications in Heterogeneous Domain Adaptation"](https://openreview.net/forum?id=m6DJxSuKuqF&noteId=SEp6zzXmpLE).

##Requirements
python3.5 <br>
scipy==1.7.1 <br>
numpy==1.20.3 <br>
matplotlib==3.4.3 <br>
cvxpy==1.2.0 <br>
pot==0.8.1.0 <br>
pytorch==1.5.0 <br>

## Instructions
The codes for HDA experiments (Section 5.2), Open-Set HDA experiments (Section 5.3), 
toy experiments for KPG-RL (Section 5.1), and toy experiments for Partial KPG-RL (Appendix B.1) are respectively in folders 
"HDA", "Open-Set HDA", "toy_data", and "toy_data_partial". Please follow the __Readme.md__ in each folder to run the codes.

## Citation:
```
@inproceedings{
gu2022keypoint,
title={Keypoint-Guided Optimal Transport with Applications in Heterogeneous Domain Adaptation},
author={Xiang Gu and Yucheng Yang and Wei Zeng and Jian Sun and Zongben Xu},
booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
year={2022}}
```