import torch
import numpy as np
import scipy.io as sio
import utils
from sklearn.svm import SVC
import ot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keypointguide_POT.linearprog import lp
from keypointguide_POT.sinkhorn import sinkhorn_log_domain

domains = ["amazon","dslr","webcam"]
num_labeled = 1
seed = 1

Tasks = []
Accs = []

for source in domains:
    for target in domains:
        print("source:{} --> target:{}".format(source, target))

        feat_s,label_s,feat_tl,label_tl,feat_tu,label_tu = utils.load_data(source,target,num_labeled,seed=seed)

        print("train svm...")
        clf = SVC(gamma='auto',probability=True)
        clf.fit(feat_tl,label_tl)
        acc = clf.score(feat_tu,label_tu)

        # print("SVM_ACC:{:.2f}".format(acc*100))
        Tasks.append(source[0].upper() + "2" + target[0].upper())
        Accs.append(round(acc * 100, 2))
Tasks.append("avg")
Accs.append(round(np.mean(np.array(Accs)), 2))
print("task:\tacc")
for k in range(len(Tasks)):
    print("{:}:\t{:.2f}".format(Tasks[k], Accs[k]))


