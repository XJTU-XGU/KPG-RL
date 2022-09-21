import numpy as np
import cvxpy as cvx
import scipy.optimize._linprog as linprog

def lp(p, q, C, Mask=None):
    c = np.reshape(C.T,(-1,1))
    b = np.vstack((p.reshape(-1,1),q.reshape(-1,1)))
    A = np.vstack((np.kron(np.ones((1,len(q))),np.eye(len(p))),
                   np.kron(np.eye(len(q)),np.ones((1,len(p))))
                   ))
    if Mask is not None:
        m = np.reshape(Mask.T,(-1,1))
        A = A*(m.T)
        c = c*m
    x = cvx.Variable((len(c),1))
    cons = [x>=0,
            A@x==b]
    obj = cvx.Minimize(c.T@x)
    prob = cvx.Problem(obj,cons)
    prob.solve()

    print(prob.status)
    pi = x.value
    pi = np.reshape(pi,(len(q),len(p)))
    pi = pi.T
    if Mask is not None:
        pi = pi*Mask
    return pi


def lp_sci(p, q, C, Mask=None):
    c = np.reshape(C.T,(-1,1))
    b = np.vstack((p.reshape(-1,1),q.reshape(-1,1)))
    A = np.vstack((np.kron(np.ones((1,len(q))),np.eye(len(p))),
                   np.kron(np.eye(len(q)),np.ones((1,len(p))))
                   ))
    if Mask is not None:
        m = np.reshape(Mask.T,(-1,1))
        A = A*(m.T)
        c = c*m

    res = linprog.linprog(c.reshape(-1,),A_eq=A,b_eq=b.reshape(-1,))
    pi = res["x"]
    pi = np.reshape(pi,(len(q),len(p)))
    pi = pi.T
    if Mask is not None:
        pi = pi*Mask
    return pi



