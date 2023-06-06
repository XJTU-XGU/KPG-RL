from .optim import cg
import torch
import time

def init_matrix(C1, C2, p, q):
    device = C1.device
    def f1(a):
        return (a**2)

    def f2(b):
        return (b**2)

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = torch.matmul(torch.matmul(f1(C1), p.view(-1, 1)),
                     torch.ones(len(q)).to(device).view(1, -1))
    constC2 = torch.matmul(torch.ones(len(p)).to(device).view(-1, 1),
                     torch.matmul(q.view(1, -1), f2(C2).t()))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    A = -torch.matmul(torch.matmul(hC1, T),(hC2.T))
    tens = constC + A
    return tens


def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2, T)
    return torch.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2,T)


def gromov_wasserstein(C1, C2, p, q, lam = 0.1, Mask = None,OT_algorithm="sinkhorn",max_iter_sinkhorn=2000,numItermax=5,fused=False,Cxy=None,alpha=0.5):

    constC, hC1, hC2 = init_matrix(C1, C2, p, q)

    G0 = p.view(-1,1) * q.view(1,-1)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)
    pi = cg(p, q, C1, C2, constC, f, df, G0, lam=lam, Mask = Mask,OT_algorithm=OT_algorithm,
            max_iter_sinkhorn=max_iter_sinkhorn,numItermax=numItermax,fused=fused,Cxy=Cxy,alpha=alpha)
    return pi

