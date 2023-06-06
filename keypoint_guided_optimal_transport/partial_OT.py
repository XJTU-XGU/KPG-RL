import torch
from . import linearprog
from . import sinkhorn


def partial_ot(p,q,C,I,J,s,xi=None,A=None,reg=0.001):
    if A is None:
        A = C.max()
    if xi is None:
        xi = 1e2*C.max()

    C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
    C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
    C_[-1, -1] = 2 * xi + A

    M = torch.ones_like(C, dtype=torch.int64)
    M[I, :] = 0
    M[:, J] = 0
    M[I, J] = 1
    a = torch.ones(M.size(0), 1, dtype=torch.int64)
    a[I] = 0
    b = torch.ones(M.size(1) + 1, 1, dtype=torch.int64)
    b[J] = 0
    M_ = torch.cat((M, a), dim=1)
    M_ = torch.cat((M_, b.t()), dim=0)

    p_ = torch.cat((p, (torch.sum(q) - s) * torch.Tensor([1])))
    q_ = torch.cat((q, (torch.sum(p) - s) * torch.Tensor([1])))

    pi_ = linearprog.lp(p_.numpy(), q_.numpy(), C_.numpy(), M_.numpy())
    # pi_ = sinkhorn.sinkhorn_log_domain(p_, q_, C_, M_,reg=reg)
    # pi = pi_[:-1, :-1]
    return pi_

def partial_ot_source(p,q,C,I,J,s,xi=None,A=None):
    if A is None:
        A = 10*C.max()
    if xi is None:
        xi = -C.max()

    C_ = torch.cat((C, xi * torch.ones(C.size(0), 1)), dim=1)
    # C_ = torch.cat((C_, xi * torch.ones(1, C_.size(1))), dim=0)
    # C_[-1, -1] = 2 * xi + A

    M = torch.ones_like(C, dtype=torch.int64)
    M[I, :] = 0
    M[:, J] = 0
    M[I, J] = 1
    a = torch.ones(M.size(0), 1, dtype=torch.int64)
    a[I] = 0
    # b = torch.ones(M.size(1) + 1, 1, dtype=torch.int64)
    # b[J] = 0
    M_ = torch.cat((M, a), dim=1)
    # M_ = torch.cat((M_, b.t()), dim=0)

    # p_ = torch.cat((p, (torch.sum(q) - s) * torch.Tensor([1])))
    q_ = torch.cat((q, (1 - s) * torch.Tensor([1])))

    pi_ = linearprog.lp(p.numpy(), q_.numpy(), C_.numpy())
    return pi_
