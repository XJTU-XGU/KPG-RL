import torch
import numpy as np

def sinkhorn_log_domain_torch(p,q,C,Mask = None, reg=0.01,niter=2000,thresh = 1e-3):

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        M =  (-C + torch.unsqueeze(u,1) + torch.unsqueeze(v,0)) / reg
        if Mask is not None:
            M[Mask==0] = -1e6
        return M

    def lse(A):
        "log-sum-exp"
        max_A,_ = torch.max(A, dim=1, keepdims=True)
        return torch.log(torch.exp(A-max_A).sum( 1, keepdims=True) + 1e-10) + max_A  # add 10^-10to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * p, 0. * q, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = reg * (torch.log(p) - lse(M(u, v)).squeeze()) + u
        v = reg * (torch.log(q) - lse(M(u, v).T).squeeze()) + v


        err = torch.sum(torch.abs(u - u1))

        actual_nits += 1
        if err < thresh:
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    print("iter:",actual_nits)
    return pi


def sinkhorn_log_domain(p,q,C,Mask = None, reg=0.1,niter=10000,thresh = 1e-5):
    C /= C.max()
    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        M =  (-C + np.expand_dims(u,1) + np.expand_dims(v,0)) / reg
        if Mask is not None:
            M[Mask==0] = -1e6
        return M

    def lse(A):
        "log-sum-exp"

        # return np.log(np.exp(A).sum(1, keepdims=True) + 1e-10)
        max_A = np.max(A, axis=1, keepdims=True)
        return np.log(np.exp(A-max_A).sum(1, keepdims=True) + 1e-10) + max_A  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * p, 0. * q, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = reg * (np.log(p) - lse(M(u, v)).squeeze()) + u
        v = reg * (np.log(q) - lse(M(u, v).T).squeeze()) + v
        err = np.linalg.norm(u - u1)

        actual_nits += 1
        if err < thresh:
            break
    U, V = u, v
    pi = np.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    return pi
