import torch
import numpy as np
import scipy.io as sio

def cost_matrix(x, y):
    # x, y = torch.Tensor(x), torch.Tensor(y)
    # Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
    Cxy = np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
    return Cxy


def guiding_matrix(C1, C2, I, J, tau_s=0.1,tau_t=0.1):
    C1_kp = C1[:, I]
    C2_kp = C2[:, J]
    Rs = softmax_matrix(-2*C1_kp / tau_s)
    Rt = softmax_matrix(-2*C2_kp / tau_t)
    S = JS_matrix(Rs, Rt)
    return S


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x

def softmax_matrix(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
    return f_x


def KL_matrix(p, q, eps=1e-10):
    return np.sum(p * np.log(p + eps) - p * np.log(q + eps), axis=-1)


def JS_matrix(P, Q, eps=1e-10):
    P = np.expand_dims(P, axis=1)
    Q = np.expand_dims(Q, axis=0)
    kl1 = KL_matrix(P, (P + Q) / 2, eps)
    kl2 = KL_matrix(Q, (P + Q) / 2, eps)
    return 0.5 * (kl1 + kl2)
