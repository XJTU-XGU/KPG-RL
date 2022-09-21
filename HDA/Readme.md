# Codes for HDA experiments
Before training, download the decaf features, resnet50 features, and sample indexes from 
[here](https://drive.google.com/drive/folders/1kSC_PFkGDWwYApZ6bHYcVBWbf1iOwN1F?usp=sharing), and 
put them "./data/decaf", "./data/resnet50", and "./data/labeled_index", respectively.



For __KPG-RL__, run

```
python KPG_RL.py
```
For __KPG (w/ dist)__, run
```
python KPG_w_dist.py
```
For the baseline __Labeled-target-only__, run
```
python train_only_labeled.py
```
