# Author: Hnikarr Bjarmi Franklínsson
# Date: 29.09.2022
# Project: Project 7
# Acknowledgements: after trying to get 1.1 for hours without success I gave up and just got it from Almar Geir Alfreðsson
# Also I got help from Almar Geir Alfreðsson in almost every section


import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal

#Section 1

#dæmi 1.1

def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    ...
    fi = []
    for i in range(features.shape[0]):
        temp = []
        for j in range(mu.shape[0]):
            temp.append(multivariate_normal.pdf(features[i], mu[j], sigma))
        fi.append(temp)
    return np.array(fi)

X, t = load_regression_iris()
N, D = X.shape

M, sigma = 10, 10
mu = np.zeros((M, D))
for i in range(D):
    mmin = np.min(X[i, :])
    mmax = np.max(X[i, :])
    mu[:, i] = np.linspace(mmin, mmax, M)
fi = mvn_basis(X, mu, sigma)


#dæmi 1.2

def _plot_mvn():
    for i in fi.T:
        plt.plot(i)
    plt.show()



#dæmi 1.3

def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    
    rows = fi.shape[1]
    ident = np.identity(rows) * lamda

    Max_w = (np.linalg.inv(fi.T.dot(fi) + ident)).dot(fi.T.dot(targets))
   
    return Max_w


#dæmi 1.4

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:

    predict = mvn_basis(features,mu,sigma)
    return predict.dot(w)
