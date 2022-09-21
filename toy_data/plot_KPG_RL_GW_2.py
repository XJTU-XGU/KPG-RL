import matplotlib.pyplot as plt
import data
import utils
from keypointguide_OT.KPG_GW import gromov_wasserstein_sinkhorn
import numpy as np
import os
import ot
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

num = 20
source_,target_ = data.get_data(num)

source = np.vstack(source_)
target = np.vstack(target_)

plt.xticks([])
plt.yticks([])
plt.tight_layout()

p = np.ones(len(source))/len(source)
q = np.ones(len(target))/len(target)
C1 = utils.cost_matrix(source,source)
C2 = utils.cost_matrix(target,target)
C1 = C1/C1.max()
C2 = C2/C2.max()
I = [1,num+1]
J = [6,num+5]
C = utils.structure_metrix_relation(C1,C2,I,J)
C = C/C.max()
M = np.ones_like(C)
M[I,:] = 0
M[:,J] = 0
M[I,J] = 1

alpha = 0.5

C1,C2,C,M,p,q = torch.Tensor(C1),torch.Tensor(C2),torch.Tensor(C),torch.Tensor(M),torch.Tensor(p),torch.Tensor(q)
pi = gromov_wasserstein_sinkhorn(C1,C2,p,q,Mask=M,OT_algorithm="lp",fused=True,Cxy=C,alpha=alpha)
source_transport = pi@target/p.reshape((-1,1))
for i in range(len(source)):
    plt.plot([source[i][0],source_transport[i][0]],[source[i][1],source_transport[i][1]],
             '-',color="grey",linewidth=0.5)


s = ["+","o","^"]
for i in range(3):
    plt.plot(source_[i][:,0], source_[i][:,1], 'b{}'.format(s[i]), markersize=10, markerfacecolor="white")
    plt.plot(target_[i][:,0], target_[i][:,1], 'g{}'.format(s[i]), markersize=10, markerfacecolor="white")
for i in range(len(I)):
    plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1],  source_transport[I[i]][1]],
             'r-', linewidth=1.0)
t = 5
for i in range(len(I)):
    plt.plot(source[I[i]][0],source[I[i]][1],'r{}'.format(s[i]), markersize=10 + t, markerfacecolor="white")
    plt.plot(source_transport[I[i]][0], source_transport[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t, markerfacecolor="white")

labels = [0]*num + [1]*num + [2]*num
labels = np.array(labels)
pred = np.argmax(pi,axis=1)
pred = labels[pred]
acc = np.mean(labels==pred)

plt.text(-3.8,-0.5,"Matching\naccuracy: {:.1f}%".format(acc*100),fontsize=22)
plt.savefig("figure/KPG_RL_GW_2.pdf")
plt.show()

