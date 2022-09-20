import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from collections import Counter
from pyds import MassFunction


def Gaussian_Distribution(N=2, M=1000, m=[0, 0], sigma=1):

    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)

    return data

num = 500

data1 = Gaussian_Distribution(N=2, M=num, m=[1, 0], sigma=0.25)
data2 = Gaussian_Distribution(N=2, M=num, m=[-1, 0], sigma=1)


data1 = pd.DataFrame(data1, columns=['x1', 'x2'], index=None)
data2 = pd.DataFrame(data2, columns=['x1', 'x2'], index=None)

data1.insert(2, 'Y', 1)
data2.insert(2, 'Y', 2)

data = pd.concat([data1, data2], axis=0)

data.to_csv('dataset.txt', index=None)