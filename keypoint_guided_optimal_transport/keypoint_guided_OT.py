from . import utils,linearprog,sinkhorn,KPG_GW,partial_OT
import numpy as np
import torch

class KeyPointGuidedOT(object):
    def __init__(self):
        super(KeyPointGuidedOT).__init__()

    def cost_matrix(self,xs,xt,cost_function="L2",eps=1e-10):
        if cost_function == "L2":
            return utils.cost_matrix(xs,xt)
        elif cost_function == "cosine":
            xs = xs / (np.linalg.norm(xs, axis=1, keepdim=True) + eps)
            xt = xt / (np.linalg.norm(xt, axis=1, keepdim=True) + eps)
            return 0.5*utils.cost_matrix(xs, xt)
        else:
            return cost_function(xs,xt)

    def kpg_rl(self,p,q,xs,xt,K,cost_function="L2",algorithm="linear_programming",tau_s=0.1,tau_t=0.1,normalized=True,
               reg=0.0001,max_iterations=100000,thres=1e-5,eps=1e-10):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        '''
        I = [tup[0] for tup in K]
        J = [tup[1] for tup in K]

        ## guiding matrix
        Cs = self.cost_matrix(xs,xs,cost_function,eps)
        Ct = self.cost_matrix(xt,xt,cost_function,eps)
        if normalized:
            Cs /= (Cs.max() + 0)
            Ct /= (Ct.max() + 0)
        G = utils.guiding_matrix(Cs,Ct,I,J,tau_s,tau_t)

        ## mask matrix
        M = np.ones_like(G)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1

        ## solving model
        if algorithm == "linear_programming":
            pi = linearprog.lp_sci(p,q,G,M)
        elif algorithm == "sinkhorn":
            pi = sinkhorn.sinkhorn_log_domain(p,q,G,M,reg,max_iterations,thres)
        else:
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        return pi

    def kpg_rl_kp(self,p,q,xs,xt,K,alpha=0.5,cost_function="L2",algorithm="linear_programming",tau_s=0.1,tau_t=0.1,
                  normalized=True,reg=0.0001,max_iterations=100000,eps=1e-10,thres=1e-5):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param alpha: float, combination factor in (0,1) for KPG-RL-KP model.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        '''
        I = [tup[0] for tup in K]
        J = [tup[1] for tup in K]

        ## guiding matrix
        Cs = self.cost_matrix(xs,xs,cost_function,eps)
        Ct = self.cost_matrix(xt,xt,cost_function,eps)
        if normalized:
            Cs /= (Cs.max() + eps)
            Ct /= (Ct.max() + eps)
        G = utils.guiding_matrix(Cs,Ct,I,J,tau_s,tau_t)

        ## cost matrix
        C = self.cost_matrix(xs,xt,cost_function,eps)
        C /= (C.max() + eps)
        G = alpha*C + (1-alpha)*G

        ## mask matrix
        M = np.ones_like(G)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1

        ## solving model
        if algorithm == "linear_programming":
            pi = linearprog.lp_sci(p,q,G,M)
        elif algorithm == "sinkhorn":
            pi = sinkhorn.sinkhorn_log_domain(p,q,G,M,reg,max_iterations,thres)
        else:
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")

        return pi

    def kpg_rl_gw(self, p, q, xs, xt, K, alpha=0.5, cost_function="L2", algorithm="linear_programming", tau_s=0.1,
                  tau_t=0.1, normalized=True, reg=0.0001, max_iterations=100000, eps=1e-10, thres=1e-5):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param alpha: float, combination factor in (0,1) for KPG-RL-KP model.
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        '''
        I = [tup[0] for tup in K]
        J = [tup[1] for tup in K]

        ## guiding matrix
        Cs = self.cost_matrix(xs, xs, cost_function, eps)
        Ct = self.cost_matrix(xt, xt, cost_function, eps)
        if normalized:
            Cs /= (Cs.max() + eps)
            Ct /= (Ct.max() + eps)
        G = utils.guiding_matrix(Cs, Ct, I, J, tau_s, tau_t)

        ## mask matrix
        M = np.ones_like(G)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1

        ## transport plan
        Cs, Ct, G, M, p, q = torch.Tensor(Cs), torch.Tensor(Ct), torch.Tensor(G), torch.Tensor(M), torch.Tensor(
            p), torch.Tensor(q)
        if algorithm != "linear_programming" and algorithm == "sinkhorn":
            raise ValueError("algorithm must be 'linear_programming' or 'sinkhorn'!")
        pi = KPG_GW.gromov_wasserstein(Cs, Ct, p, q, Mask=M, OT_algorithm=algorithm, fused=True, Cxy=G, alpha=alpha)
        return pi

    def partial_kpg_rl(self, p, q, xs, xt, K, s=0.5, cost_function="L2", tau_s=1.0,
                  tau_t=1.0, normalized=False,eps=1e-10):
        '''
        :param p: ndarray, (m,), Mass of source samples
        :param q: ndarray, (n,), Mass of target samples
        :param xs: ndarray, (m,d), d-dimensional source samples
        :param xt: ndarray, (n,d), d-dimensional target samples
        :param K: list of tuples, e.g., [(0,1),(10,20)]. Each tuple is an index pair of keypoints.
        :param s: float, total transported mass in (0,1).
        :param cost_function: str or function, type of cost function. Default is "L2". Choices should be "L2", "cosine",
        and a pre-defined function.
        :param algorithm: str, algorithm to solve model. Default is "linear_programming". Choices should be
        "linear_programming" and "sinkhorn".
        :param tau_s: float, source temperature for computing the relation.
        :param tau_t: float, target temperature for computing the relation.
        :param normalized: bool, whether to normalize the distance
        :param reg: float, regularization coefficient in entropic model
        :param max_iterations: int, maximum number of iterations
        :param eps: float, a small number to avoid NaN
        :param thres: float, stop criterion for sinkhorn
        :return: transport plan, (m,n)
        '''
        I = [tup[0] for tup in K]
        J = [tup[1] for tup in K]

        ## guiding matrix
        Cs = self.cost_matrix(xs, xs, cost_function, eps)
        Ct = self.cost_matrix(xt, xt, cost_function, eps)
        if normalized:
            Cs /= (Cs.max() + eps)
            Ct /= (Ct.max() + eps)
        G = utils.guiding_matrix(Cs, Ct, I, J, tau_s, tau_t)

        ## mask matrix
        M = np.ones_like(G)
        M[I, :] = 0
        M[:, J] = 0
        M[I, J] = 1

        ## transport plan
        pi = partial_OT.partial_ot(torch.Tensor(p), torch.Tensor(q), torch.Tensor(G), I, J, s=s)
        return pi[:-1,:-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #
    # ############################################################################################################
    # ##### Example for KPG-RL
    # def get_data(num=10):
    #     np.random.seed(3)
    #     data0 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
    #     data1 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
    #     data2 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
    #     data3 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
    #     source = np.vstack((data0, data1 - [0, 3]))
    #     target = np.vstack((data2 - [1.5, 1.5], data3 - [-1.5, 1.5]))
    #     return source, target
    #
    # num = 10
    # source, target = get_data(num)
    # source_positive = source[:num]
    # source_negative = source[num:]
    # target_positive = target[:num]
    # target_negative = target[num:]
    #
    # m, n = len(source), len(target)
    #
    # xs = source
    # xt = target
    # p = np.ones(m) / m
    # q = np.ones(n) / n
    # K = [(0,0),(num+1,num+1)]
    #
    # kgot = KeyPointGuidedOT()
    # pi = kgot.kpg_rl(p,q,xs,xt,K)
    # # pi = kgot.kpg_rl(p, q, xs, xt, K,
    # #                  algorithm="sinkhorn",
    # #                  reg=1e-6,
    # #                  thres=1e-20)
    #
    # ### show transport plan
    # I = [tup[0] for tup in K]
    # J = [tup[1] for tup in K]
    # source_transport = pi @ target / p.reshape((-1, 1))
    # plt.figure()
    # for i in range(len(source)):
    #     plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
    #              '-', color="grey", linewidth=1.0)
    # for i in range(len(I)):
    #     plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1], source_transport[I[i]][1]],
    #              'r-', linewidth=1.0)
    # s = 5
    # plt.plot(source_positive[:, 0], source_positive[:, 1], 'b+', markersize=12)
    # plt.plot(source_negative[:, 0], source_negative[:, 1], 'bo', markersize=10, markerfacecolor="white")
    # plt.plot(target_positive[:, 0], target_positive[:, 1], 'g+', markersize=12)
    # plt.plot(target_negative[:, 0], target_negative[:, 1], 'go', markersize=10, markerfacecolor="white")
    # plt.plot(source[I[0]][0], source[I[0]][1], 'r+', markersize=10 + s, linewidth=3)
    # plt.plot(source[I[1]][0], source[I[1]][1], 'ro', markersize=8 + s, markerfacecolor="white", linewidth=3)
    # plt.plot(source_transport[I[0]][0], source_transport[I[0]][1], 'r+', markersize=10 + s)
    # plt.plot(source_transport[I[1]][0], source_transport[I[1]][1], 'ro', markersize=8 + s, markerfacecolor="white")
    # plt.xlim([-2.2, 1.9])
    # plt.ylim([-2.7, 1.6])
    # plt.xticks([])
    # plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    #
    # ###############################################################################################################
    # ##### Example for KPG-RL-KP
    # num=20
    # def get_data(num=20):
    #     np.random.seed(3)
    #     source = []
    #     target = []
    #     centers = [np.array([[-1, -1]]), np.array([[-3, 2]]), np.array([[-2, 3]]),
    #                np.array([[0, 1]]), np.array([[-0.5, 0.5]]), np.array([[-1, 2]])]
    #     for i in range(3):
    #         source.append(
    #             np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) +
    #             centers[i])
    #         target.append(
    #             np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) +
    #             centers[
    #                 i + 3])
    #     return source, target
    #
    #
    # source_, target_ = get_data(num)
    #
    # source = np.vstack(source_)
    # target = np.vstack(target_)
    # p = np.ones(len(source)) / len(source)
    # q = np.ones(len(target)) / len(target)
    # xs,xt = source,target
    # K = [(3,8),(num+1,num+1),(2*num+6,2*num+13)]
    #
    # kgot = KeyPointGuidedOT()
    # pi = kgot.kpg_rl_kp(p, q, xs, xt, K)
    # I = [tup[0] for tup in K]
    # J = [tup[1] for tup in K]
    #
    # source_transport = pi @ target / p.reshape((-1, 1))
    # for i in range(len(source)):
    #     plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
    #              '-', color="black", linewidth=0.5)
    # s = ["+", "o", "^"]
    # for i in range(3):
    #     plt.plot(source_[i][:, 0], source_[i][:, 1], 'b{}'.format(s[i]), markersize=10, markerfacecolor="white")
    #     plt.plot(target_[i][:, 0], target_[i][:, 1], 'g{}'.format(s[i]), markersize=10, markerfacecolor="white")
    # for i in range(len(I)):
    #     plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1], source_transport[I[i]][1]],
    #              'r-', linewidth=1.0)
    # t = 5
    # for i in range(len(I)):
    #     plt.plot(source[I[i]][0], source[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t, markerfacecolor="white")
    #     plt.plot(source_transport[I[i]][0], source_transport[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t,
    #              markerfacecolor="white")
    #
    # labels = [0] * num + [1] * num + [2] * num
    # labels = np.array(labels)
    # pred = np.argmax(pi, axis=1)
    # pred = labels[pred]
    # acc = np.mean(labels == pred)
    #
    # plt.text(-3.8, -0.5, "Matching\naccuracy: {:.1f}%".format(acc * 100), fontsize=22)
    # plt.show()

    ###############################################################################################################
    ##### Example for Partial-KPG-RL
    def get_data(num=20, n_modes=3):
        np.random.seed(5)
        source = []
        target = []
        centers = [np.array([[-2, 0]]), np.array([[-3, 1]]), np.array([[-2, 3]]),
                   np.array([[-1, 1.5]]), np.array([[-2, 1]]), np.array([[-1, 2]])]
        for i in range(n_modes):
            source.append(
                np.random.multivariate_normal(np.array([0, 0]), cov=0.03 * np.array([[1, 0], [0, 1]]), size=num) +
                centers[i])
            if i < 2:
                target.append(
                    np.random.multivariate_normal(np.array([0, 0]), cov=0.03 * np.array([[1, 0], [0, 1]]), size=num) +
                    centers[
                        i + 3])
        return source, target


    num = 10
    source_, target_ = get_data(num)

    source = np.vstack(source_)
    target = np.vstack(target_)

    p = np.ones(len(source)) / len(source)
    q = np.ones(len(target))*p[0]
    xs,xt = source,target
    K = [(3,3),(num+2,num+5)]
    s = 2/3

    kgot = KeyPointGuidedOT()
    pi = kgot.partial_kpg_rl(p, q, xs, xt, K,s=s)

    I = [tup[0] for tup in K]
    J = [tup[1] for tup in K]
    selected_index = np.argwhere(np.sum(pi,axis=1) > 1e-4)
    source_transport = pi @ target / np.sum(pi, axis=1, keepdims=True)

    for i in range(len(source)):
        if i in selected_index:
            plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
                     '-', color="black", linewidth=0.5)
    s = ["+", "o", "^"]
    for i in range(3):
        plt.plot(source_[i][:, 0], source_[i][:, 1], 'b{}'.format(s[i]), markersize=10, markerfacecolor="white")
        if i <= len(target_) - 1:
            plt.plot(target_[i][:, 0], target_[i][:, 1], 'g{}'.format(s[i]), markersize=10, markerfacecolor="white")
    for i in range(len(I)):
        plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1], source_transport[I[i]][1]],
                 'r-', linewidth=1.0)
    t = 5
    for i in range(len(I)):
        plt.plot(source[I[i]][0], source[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t, markerfacecolor="white")
        plt.plot(source_transport[I[i]][0], source_transport[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t,
                 markerfacecolor="white")
    plt.show()

