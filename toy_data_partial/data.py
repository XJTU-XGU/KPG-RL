import numpy as np
import matplotlib.pyplot as plt

def get_data(num=20,n_modes = 3):
    np.random.seed(5)
    source = []
    target = []
    centers = [np.array([[-2,0]]),np.array([[-3,1]]),np.array([[-2,3]]),
               np.array([[-1,1.5]]),np.array([[-2,1]]),np.array([[-1,2]])]
    for i in range(n_modes):
        source.append(np.random.multivariate_normal(np.array([0,0]),cov=0.03*np.array([[1,0],[0,1]]),size=num)+centers[i])
        if i <2:
            target.append(
                np.random.multivariate_normal(np.array([0, 0]), cov=0.03 * np.array([[1, 0], [0, 1]]), size=num) + centers[
                    i+3])
    return source,target

