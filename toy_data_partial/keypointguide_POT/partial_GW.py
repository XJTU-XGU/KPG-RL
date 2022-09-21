from .partial_optim import cg
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

def guide_loss(C1,C2, T,I,J):
    loss = 0
    for m in range(len(I)):
        i = I[m]
        j = J[m]
        loss = loss + torch.sum(T*(C1[i].view(-1,1)-C2[j].view(1,-1))**2)
    return loss

def guide_loss_grad(C1,C2, T,I,J):
    T_ = 1.0*T
    T_.requires_grad = True
    loss = guide_loss(C1,C2, T_,I,J)
    grad = torch.autograd.grad(loss,T_)[0].data
    return grad

def gwggrad(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2,T)


def partial_GW(C1, C2, p, q, I,J,s=0.5,xi=None,A=0,reg=0.1,numItermax=1):

    constC, hC1, hC2 = init_matrix(C1, C2, p, q)

    G0 = p.view(-1,1) * q.view(1,-1)

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)
    t1 = time.time()
    pi,pi_ = cg(p, q, C1, C2, I, J, s, constC, f, df, G0, xi, A, numItermax=numItermax)
    t2 = time.time()
    # print("cost time:",t2-t1)
    return pi,pi_
