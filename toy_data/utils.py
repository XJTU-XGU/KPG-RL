import torch
import numpy as np
import scipy.io as sio
import tqdm

def cost_matrix(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
    # Cxy = np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
    return Cxy.numpy()


def structure_metrix(C1, C2, I, J):
    S = np.zeros((len(C1), len(C2)))
    for m in range(len(I)):
        i = I[m]
        j = J[m]
        S = S + (C1[i].reshape(-1, 1) - C2[j].reshape(1, -1)) ** 2
    return S


def structure_metrix_relation(C1, C2, I, J, tau=0.1):
    # print("get structure data...")
    S = np.zeros((len(C1), len(C2)))
    C1_kp = C1[:, I]
    C2_kp = C2[:, J]
    R1 = softmax_matrix(-2*C1_kp / tau)
    R2 = softmax_matrix(-2*C2_kp / tau)
    S = JS_matrix(R1, R2)
    return S


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)))
    return f_x


def KL(p, q, eps=1e-10):
    return np.sum(p * np.log(p + eps) - p * np.log(q + eps))


def JS(p, q, eps=1e-10):
    return 0.5 * KL(p, (p + q) / 2, eps) + 0.5 * KL(q, (p + q) / 2, eps)


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
