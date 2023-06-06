from keypoint_guided_optimal_transport.keypoint_guided_OT import KeyPointGuidedOT
import numpy as np
import matplotlib.pyplot as plt

def example_of_kpg_rl():
    def get_data(num=10):
        np.random.seed(3)
        data0 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
        data1 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
        data2 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
        data3 = np.random.multivariate_normal(np.array([0, 1]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num)
        source = np.vstack((data0, data1 - [0, 3]))
        target = np.vstack((data2 - [1.5, 1.5], data3 - [-1.5, 1.5]))
        return source, target

    num = 10
    source, target = get_data(num)
    source_positive = source[:num]
    source_negative = source[num:]
    target_positive = target[:num]
    target_negative = target[num:]

    m, n = len(source), len(target)

    xs = source
    xt = target
    p = np.ones(m) / m
    q = np.ones(n) / n
    K = [(0,0),(num+1,num+1)]

    kgot = KeyPointGuidedOT()
    pi = kgot.kpg_rl(p,q,xs,xt,K)
    # pi = kgot.kpg_rl(p, q, xs, xt, K,
    #                  algorithm="sinkhorn",
    #                  reg=1e-6,
    #                  thres=1e-20)

    ### show transport plan
    I = [tup[0] for tup in K]
    J = [tup[1] for tup in K]
    source_transport = pi @ target / p.reshape((-1, 1))
    plt.figure()
    for i in range(len(source)):
        plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
                 '-', color="grey", linewidth=1.0)
    for i in range(len(I)):
        plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1], source_transport[I[i]][1]],
                 'r-', linewidth=1.0)
    s = 5
    plt.plot(source_positive[:, 0], source_positive[:, 1], 'b+', markersize=12)
    plt.plot(source_negative[:, 0], source_negative[:, 1], 'bo', markersize=10, markerfacecolor="white")
    plt.plot(target_positive[:, 0], target_positive[:, 1], 'g+', markersize=12)
    plt.plot(target_negative[:, 0], target_negative[:, 1], 'go', markersize=10, markerfacecolor="white")
    plt.plot(source[I[0]][0], source[I[0]][1], 'r+', markersize=10 + s, linewidth=3)
    plt.plot(source[I[1]][0], source[I[1]][1], 'ro', markersize=8 + s, markerfacecolor="white", linewidth=3)
    plt.plot(source_transport[I[0]][0], source_transport[I[0]][1], 'r+', markersize=10 + s)
    plt.plot(source_transport[I[1]][0], source_transport[I[1]][1], 'ro', markersize=8 + s, markerfacecolor="white")
    plt.xlim([-2.2, 1.9])
    plt.ylim([-2.7, 1.6])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def example_of_kpg_rl_kp():
    num=20
    def get_data(num=20):
        np.random.seed(3)
        source = []
        target = []
        centers = [np.array([[-1, -1]]), np.array([[-3, 2]]), np.array([[-2, 3]]),
                   np.array([[0, 1]]), np.array([[-0.5, 0.5]]), np.array([[-1, 2]])]
        for i in range(3):
            source.append(
                np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) +
                centers[i])
            target.append(
                np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) +
                centers[
                    i + 3])
        return source, target


    source_, target_ = get_data(num)

    source = np.vstack(source_)
    target = np.vstack(target_)
    p = np.ones(len(source)) / len(source)
    q = np.ones(len(target)) / len(target)
    xs,xt = source,target
    K = [(3,8),(num+1,num+1),(2*num+6,2*num+13)]

    kgot = KeyPointGuidedOT()
    pi = kgot.kpg_rl_kp(p, q, xs, xt, K)
    I = [tup[0] for tup in K]
    J = [tup[1] for tup in K]

    source_transport = pi @ target / p.reshape((-1, 1))
    for i in range(len(source)):
        plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
                 '-', color="black", linewidth=0.5)
    s = ["+", "o", "^"]
    for i in range(3):
        plt.plot(source_[i][:, 0], source_[i][:, 1], 'b{}'.format(s[i]), markersize=10, markerfacecolor="white")
        plt.plot(target_[i][:, 0], target_[i][:, 1], 'g{}'.format(s[i]), markersize=10, markerfacecolor="white")
    for i in range(len(I)):
        plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1], source_transport[I[i]][1]],
                 'r-', linewidth=1.0)
    t = 5
    for i in range(len(I)):
        plt.plot(source[I[i]][0], source[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t, markerfacecolor="white")
        plt.plot(source_transport[I[i]][0], source_transport[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t,
                 markerfacecolor="white")

    labels = [0] * num + [1] * num + [2] * num
    labels = np.array(labels)
    pred = np.argmax(pi, axis=1)
    pred = labels[pred]
    acc = np.mean(labels == pred)

    plt.text(-3.8, -0.5, "Matching\naccuracy: {:.1f}%".format(acc * 100), fontsize=22)
    plt.show()

def example_of_kpg_rl_gw():
    num = 20

    def get_data(num=20):
        np.random.seed(3)
        source = []
        target = []
        centers = [np.array([[-1, -1]]), np.array([[-3, 2]]), np.array([[-2, 3]]),
                   np.array([[0, 1]]), np.array([[-0.5, 0.5]]), np.array([[-1, 2]])]
        for i in range(3):
            source.append(
                np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) +
                centers[i])
            target.append(
                np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) +
                centers[
                    i + 3])
        return source, target

    source_, target_ = get_data(num)

    source = np.vstack(source_)
    target = np.vstack(target_)
    p = np.ones(len(source)) / len(source)
    q = np.ones(len(target)) / len(target)
    xs, xt = source, target
    K = [(3, 8), (num + 1, num + 1), (2 * num + 6, 2 * num + 13)]

    kgot = KeyPointGuidedOT()
    pi = kgot.kpg_rl_gw(p, q, xs, xt, K,alpha=0.5)

    I = [tup[0] for tup in K]
    J = [tup[1] for tup in K]

    source_transport = pi @ target / p.reshape((-1, 1))
    for i in range(len(source)):
        plt.plot([source[i][0], source_transport[i][0]], [source[i][1], source_transport[i][1]],
                 '-', color="black", linewidth=0.5)
    s = ["+", "o", "^"]
    for i in range(3):
        plt.plot(source_[i][:, 0], source_[i][:, 1], 'b{}'.format(s[i]), markersize=10, markerfacecolor="white")
        plt.plot(target_[i][:, 0], target_[i][:, 1], 'g{}'.format(s[i]), markersize=10, markerfacecolor="white")
    for i in range(len(I)):
        plt.plot([source[I[i]][0], source_transport[I[i]][0]], [source[I[i]][1], source_transport[I[i]][1]],
                 'r-', linewidth=1.0)
    t = 5
    for i in range(len(I)):
        plt.plot(source[I[i]][0], source[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t, markerfacecolor="white")
        plt.plot(source_transport[I[i]][0], source_transport[I[i]][1], 'r{}'.format(s[i]), markersize=10 + t,
                 markerfacecolor="white")

    labels = [0] * num + [1] * num + [2] * num
    labels = np.array(labels)
    pred = np.argmax(pi, axis=1)
    pred = labels[pred]
    acc = np.mean(labels == pred)

    plt.text(-3.8, -0.5, "Matching\naccuracy: {:.1f}%".format(acc * 100), fontsize=22)
    plt.show()

def example_of_partial_kpg_rl():
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

if __name__ == "__main__":
    example_of_kpg_rl()
    example_of_kpg_rl_kp()
    example_of_kpg_rl_gw()
    example_of_partial_kpg_rl()
