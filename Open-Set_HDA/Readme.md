# Codes for Open-Set HDA experiments
Before training, download the decaf features, resnet50 features, and sample indexes from 
[here](https://drive.google.com/drive/folders/1kSC_PFkGDWwYApZ6bHYcVBWbf1iOwN1F?usp=sharing) 
(data is the same as those for HDA), and 
put them "./data/decaf", "./data/resnet50", and "./data/labeled_index", respectively.



For __Partial-KPG-RL__, run

```
python KPG_RL.py
```

For the baseline, run
```
python train_only_labeled.py
```
