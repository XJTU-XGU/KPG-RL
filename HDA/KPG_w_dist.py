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

        ####key point
        I = []
        J = []
        t = 0
        feat_sl = []
        for l in label_tl:
            I.append(t)
            J.append(t)
            fl = feat_s[label_s==l]
            feat_sl.append(np.mean(fl,axis=0))
            t += 1
        feat_sl = np.vstack(feat_sl)
        feat_s_ = np.vstack((feat_sl,feat_s))
        feat_t_ = np.vstack((feat_tl,feat_tu))
        Cs = utils.cost_matrix(feat_s_,feat_s_)
        Cs /= Cs.max()
        Ct = utils.cost_matrix(feat_t_,feat_t_)
        Ct /= Ct.max()
        p = np.ones(len(Cs))/len(Cs)
        q = np.ones(len(Ct))/len(Ct)
        C = utils.structure_metrix(Cs,Ct,I,J)
        C = C/C.max()
        ###mask
        M = np.ones_like(C)
        M[I,:] = 0
        M[:,J] = 0
        M[I,J] = 1

        print("solving kpg-ot...")
        pi = lp(p,q,C,M)
        # pi = sinkhorn_log_domain(p,q,C,M,reg=0.005)
        feat_s_trans = pi@feat_t_/p.reshape(-1,1)
        feat_train = np.vstack((feat_tl,feat_s_trans[len(feat_tl):]))
        label_train = np.hstack((label_tl,label_s))

        print("train svm...")
        clf = SVC(gamma='auto',probability=True)
        clf.fit(feat_train,label_train)
        acc = clf.score(feat_tu,label_tu)

        # print("SVM_ACC:{:.2f}".format(acc*100))
        Tasks.append(source[0].upper() + "2" + target[0].upper())
        Accs.append(round(acc * 100, 2))
Tasks.append("avg")
Accs.append(round(np.mean(np.array(Accs)), 2))
print("task:\tacc")
for k in range(len(Tasks)):
    print("{:}:\t{:.2f}".format(Tasks[k], Accs[k]))


