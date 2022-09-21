from keypointguide_OT.sinkhorn import sinkhorn_log_domain
from .linearprog import lp
import torch

def cg(a, b, C1, C2, constC, f, df,  G0=None, lam = 0.1, Mask=None, OT_algorithm="sinkhorn",max_iter_sinkhorn=1000,numItermax=5, numItermaxEmd=100,
       stopThr=1e-4, stopThr2=1e-4,fused=False,Cxy=None,alpha=0.5):

    loop = 1

    G = G0


    def cost(G):
        if not fused:
            return f(G)
        else:
            return  alpha*f(G) + (1.0-alpha)*torch.sum(Mask*Cxy*G)

    def cost_mask(G):
        return cost(Mask*G)

    if Mask is None:
        f_val = cost(G)
    else:
        f_val = cost_mask(G)

    it = 0

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        if Mask is None:
            dfG = df(G)
        else:
            dfG = df(Mask*G)

        if fused:
            Mi = alpha*dfG + (1.0-alpha)*Mask*Cxy
        else:
            Mi = dfG

        # set M positive
        Mi += Mi.min()

        # solve OT
        if OT_algorithm == "sinkhorn":
            if Mask is None:
                Gc = sinkhorn_log_domain(a,b,Mi,reg=lam,niter=max_iter_sinkhorn)
            else:
                Gc = sinkhorn_log_domain(a, b, Mi, reg=lam, Mask=Mask,niter=max_iter_sinkhorn)
        if OT_algorithm == "lp":
            if Mask is None:
                Gc = lp(a.numpy(),b.numpy(),Mi.numpy())
            else:
                Gc = torch.Tensor(lp(a.numpy(),b.numpy(),Mi.numpy(),Mask=Mask.numpy()))

        deltaG = Gc - G

        # line search
        if Mask is None:
            alpha1, fc, f_val = solve_linesearch(cost, G, deltaG, C1,C2,constC)
        else:
            alpha1, fc, f_val = solve_linesearch(cost, Mask*G, Mask*deltaG,C1,C2,constC)

        G = G + alpha1 * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

    return G


def solve_linesearch(cost, G, deltaG, C1=None, C2=None, constC=None):

    dot1 = torch.matmul(C1, deltaG)
    dot12 = torch.matmul(dot1,C2)
    a = -2 * torch.sum(dot12 * deltaG)
    b = torch.sum(constC * deltaG) - 2 * (torch.sum(dot12 * G) + torch.sum(torch.matmul(torch.matmul(C1, G),C2) * deltaG))
    c = cost(G)

    alpha = solve_1d_linesearch_quad(a, b, c)
    fc = None
    f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val

def solve_1d_linesearch_quad(a, b, c):
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1, max(0, -b/2.0 * a))
        return minimum
    else:  # non convex
        if f0 > f1:
            return 1
        else:
            return 0
