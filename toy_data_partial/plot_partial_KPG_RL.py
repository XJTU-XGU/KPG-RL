import matplotlib.pyplot as plt
import data
import utils
from keypointguide_POT.linearprog import lp
import numpy as np
import os
import ot
import torch
from keypointguide_POT import partial_OT
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

num = 10
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

'''
As the partial OT removes the outliers with larger distances to keypoints in source domain, we normalize the target 
domain data distances to [0,0.1], and the source domain ones to [0,1], empirically increasing the effectiveness of 
partial OT for detecting outliers.
'''
C1 = C1/C1.max()
C2 = 0.1*C2/C2.max()

I = [3,num+2]
J = [3,num+5]
C = utils.structure_metrix_relation(C1,C2,I,J,tau=0.1)

C = C/C.max()
M = np.ones_like(C)
M[I,:] = 0
M[:,J] = 0
M[I,J] = 1

s = 2/3
pi_ = partial_OT.partial_ot_source(torch.Tensor(p),s*torch.Tensor(q),torch.Tensor(C),I,J,s=2/3)
pi = pi_[:,:-1]
selected_index = np.argwhere(pi_[:,-1]<1e-4)
source_transport = pi@target/np.sum(pi,axis=1,keepdims=True)
for i in range(len(source)):
    if i in selected_index:
        plt.plot([source[i][0],source_transport[i][0]],[source[i][1],source_transport[i][1]],
                 '-',color="grey",linewidth=0.5)


s = ["+","o","^"]
for i in range(3):
    plt.plot(source_[i][:,0], source_[i][:,1], 'b{}'.format(s[i]), markersize=10, markerfacecolor="white")
    if i<= len(target_)-1:
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
pred = np.argmax(pi_,axis=1)
pred = labels[pred]
acc = np.mean(labels==pred)
print(acc)
plt.text(-3.3,2,"Matching\naccuracy: {:.1f}%".format(acc*100),fontsize=22)
plt.savefig("figure/partial_KPG.pdf")
plt.show()

