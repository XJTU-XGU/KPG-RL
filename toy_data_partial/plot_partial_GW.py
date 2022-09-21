import matplotlib.pyplot as plt
import data
import utils
from keypointguide_POT.linearprog import lp
import numpy as np
import os
import ot
import torch
from keypointguide_POT import partial_OT,partial_GW
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
I = [3,num+2]
J = [3,num+5]
s = 2/3
_,pi_ = partial_GW.partial_GW(torch.Tensor(C1),torch.Tensor(C2),torch.Tensor(p),s*torch.Tensor(q),I,J,s=2/3)
pi_ = pi_.numpy()
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
t = 5
labels = [0]*num + [1]*num + [2]*num
labels = np.array(labels)
pred = np.argmax(pi_,axis=1)
pred = labels[pred]
acc = np.mean(labels==pred)
print(acc)
plt.text(-3.3,2,"Matching\naccuracy: {:.1f}%".format(acc*100),fontsize=22)
plt.savefig("figure/partial_GW.pdf")
plt.show()

