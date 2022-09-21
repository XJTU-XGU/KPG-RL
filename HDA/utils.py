import torch
import numpy as np
import scipy.io as sio
import tqdm
from sklearn.metrics import confusion_matrix


def load_data(source, target, num_labeled=1, class_balance=True, num=10, seed=0):
    source_data = sio.loadmat("./data/decaf/{}_fc6.mat".format(source))
    target_data = sio.loadmat("./data/resnet50/{}.mat".format(target))

    source_feat, source_label = source_data["fts"], source_data["labels"]
    target_feat, target_label = target_data["fts"], target_data["labels"]
    source_label, target_label = source_label.reshape(-1, ), target_label.reshape(-1, )

    indexes = sio.loadmat("./data/labeled_index/{}_{}.mat".format(target,num_labeled))
    idx_labeled,idx_unlabeled = indexes["labeled_index"][0], indexes["unlabeled_index"][0]

    target_feat_l, target_label_l = target_feat[idx_labeled], target_label[idx_labeled]
    target_feat_un, target_label_un = target_feat[idx_unlabeled], target_label[idx_unlabeled]

    if class_balance:
        source_feat, source_label = class_balancing(source_feat, source_label, num=num,seed=seed)
        target_feat_un, target_label_un = class_balancing(target_feat_un, target_label_un, num=num)

    return source_feat, source_label, target_feat_l, target_label_l, target_feat_un, target_label_un


def class_balancing(x, t, num=10, seed=0):
    t = np.squeeze(t)
    x_class = []
    n_class = t.max()
    for k in range(1, n_class + 1, 1):
        x_class.append(x[t == k])
    np.random.seed(seed)
    x_class_balance = [xx[np.random.choice(np.arange(len(xx)), num, replace=True)] for xx in x_class]
    t_class_balance = [np.ones(num, dtype=np.int32) * k for k in range(1, n_class + 1, 1)]
    x_class_balance = np.vstack(x_class_balance)
    t_class_balance = np.hstack(t_class_balance)
    return x_class_balance, t_class_balance


def cost_matrix(x, y):
    x, y = torch.Tensor(x), torch.Tensor(y)
    Cxy = x.pow(2).sum(dim=1).unsqueeze(1) + y.pow(2).sum(dim=1).unsqueeze(0) - 2 * torch.matmul(x, y.t())
    return Cxy.numpy()


def structure_metrix(C1, C2, I, J):
    S = np.zeros((len(C1), len(C2)))
    for m in range(len(I)):
        i = I[m]
        j = J[m]
        S = S + (C1[i].reshape(-1, 1) - C2[j].reshape(1, -1)) ** 2
    return S


def structure_metrix_relation(C1, C2, I, J, tau=0.1):
    C1_kp = C1[:, I]
    C2_kp = C2[:, J]
    R1 = softmax_matrix(-C1_kp / tau)
    R2 = softmax_matrix(-C2_kp / tau)
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