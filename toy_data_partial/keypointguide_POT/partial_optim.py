import torch
from . import partial_OT

def cg(a, b, C1, C2,  I,J,s, constC, f, df, G0=None, xi=None, A=None, numItermax=5,
       stopThr=1e-5, stopThr2=1e-5):

    loop = 1

    G = G0
    M = torch.ones(len(a),len(b), dtype=torch.int64)
    # M[I, :] = 1
    # M[:, J] = 1
    # M[I, J] = 1

    def cost(G):
        return f(G)

    def cost_mask(G):
        return cost(M*G)

    f_val = cost_mask(G)

    it = 0

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        dfG = df(M*G)

        Mi = dfG

        # set M positive
        Mi -= Mi.min()

        # solve OT
        pi_ = partial_OT.partial_ot_source(a,b,Mi,I,J,s,xi,A)
        pi_ = torch.Tensor(pi_)
        Gc = pi_[:, :-1]
        deltaG = Gc - G

        # line search
        alpha, fc, f_val = solve_linesearch(cost, M*G, M*deltaG,C1,C2,constC)

        if it == 1:
            G = G +  1*deltaG
        else:
            G = G +  alpha*deltaG

        # test convergence
        if it >= numItermax:
            loop = 0
        # print(f_val)

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

    return G,pi_


def solve_linesearch(cost, G, deltaG, C1=None, C2=None, constC=None,epsilon=0.1):

    dot1 = torch.matmul(C1, deltaG)
    dot12 = torch.matmul(dot1,C2)
    a = -2 * torch.sum(dot12 * deltaG)
    b = torch.sum(constC * deltaG) - 2 * (torch.sum(dot12 * G) + torch.sum(torch.matmul(torch.matmul(C1, G),C2) * deltaG))
    c = cost(G)

    alpha = solve_1d_linesearch_quad(a, b, c,epsilon)
    fc = None
    f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val

def solve_1d_linesearch_quad(a, b, c, epsilon=0.0):
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1-epsilon, max(0+epsilon, -b/2.0 * a))
        return minimum
    else:  # non convex
        if f0 > f1:
            return 1- epsilon
        else:
            return 0+epsilon
