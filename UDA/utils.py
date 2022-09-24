import torch
import numpy as np
import scipy.io as sio
import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def structure_metrix_relation(C1_kp,C2_kp,tau=0.01):
    # C1_kp = C1[:,I]
    # C2_kp = C2[:,J]
    R1 = F.softmax(-C1_kp/tau,dim=1)
    R2 = F.softmax(-C2_kp/tau,dim=1)
    S = JS_matrix(R1,R2)
    return S


def KL_matrix(p,q,eps=1e-10):
    return torch.sum(p*torch.log(p+eps)-p*torch.log(q+eps),dim=-1)


def JS_matrix(P,Q,eps=1e-10):
    P = P.unsqueeze(1)
    Q = Q.unsqueeze(0)
    kl1 = KL_matrix(P,(P+Q)/2,eps)
    kl2 = KL_matrix(Q,(P+Q)/2,eps)
    return 0.5* (kl1+kl2)