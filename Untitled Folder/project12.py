# Author: Hnikarr Bjarmi Franklínsson
# Date: 06.11.2022
# Project: Project 12
# Acknowledgements: got a lot of help from Oliver Aron Jóhannesson and just mostly copied his functions


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:

    scaler = StandardScaler()
    out = scaler.fit_transform(X)
    return out


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):

    X = standardize(X[:,[i,j]])
    plt.scatter(X[:,0],X[:,1],s=4)



def _scatter_cancer():
    X, y = load_cancer()
    X, y = load_cancer()
    for i in range(X.shape[1]):
        plt.subplot(5,6,i+1)
        plt.scatter(X[:,0],X[:,i],s=4)
    plt.show()


def _plot_pca_components():
    ...
    X, y = load_cancer()
    pca = PCA()
    X=standardize(X)
    pca.fit_transform(X)
    comps =pca.components_
    for i in range(comps.shape[1]):
        plt.subplot(5, 6, i+1)
        plt.plot(comps[i])
        plt.title('PCA '+str(i+1))
    plt.show()

def _plot_eigen_values():
    X, y = load_cancer()
    pca = PCA()
    X=standardize(X)
    pca.fit_transform(X)
    plt.plot(pca.explained_variance_)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()



def _plot_log_eigen_values():
    X, y = load_cancer()
    pca = PCA()
    X=standardize(X)
    pca.fit_transform(X)
    plt.plot(np.log(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    X, y = load_cancer()
    pca = PCA()
    X=standardize(X)
    pca.fit_transform(X)
    plt.plot(np.cumsum(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()


_plot_cum_variance()