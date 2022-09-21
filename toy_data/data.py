import numpy as np
import matplotlib.pyplot as plt

def get_data(num=20):
    np.random.seed(3)
    source = []
    target = []
    centers = [np.array([[-1,-1]]),np.array([[-3,2]]),np.array([[-2,3]]),
               np.array([[0,1]]),np.array([[-0.5,0.5]]),np.array([[-1,2]])]
    for i in range(3):
        source.append(np.random.multivariate_normal(np.array([0,0]),cov=0.05*np.array([[1,0],[0,1]]),size=num)+centers[i])
        target.append(
            np.random.multivariate_normal(np.array([0, 0]), cov=0.05 * np.array([[1, 0], [0, 1]]), size=num) + centers[
                i+3])
    return source,target

