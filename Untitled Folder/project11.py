# Author: Hnikarr Bjarmi Franklínsson
# Date: 06.11.2022
# Project: Project 11
# Acknowledgements: got a lot of help from Oliver Aron Jóhannesson and just mostly copied his functions


import numpy as np
import sklearn as sk
from scipy.spatial import distance 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results

def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:

    n, f = X.shape
    k    = Mu.shape[0]
    out = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            out[i,j] = distance.euclidean(X[i,:],Mu[j,:])
    return out


def determine_r(dist: np.ndarray) -> np.ndarray:

    out = np.zeros_like(dist)
    n,k = dist.shape
    for i in range(dist.shape[0]):
        result = np.where(dist[i] == np.amin(dist[i]))
        out[i,result[0]] = 1
    return out


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:

    n,k = R.shape
    return 1/n*sum(sum(R*dist))


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:

    K, F = Mu.shape
    N, F = X.shape
    new_mu = np.zeros_like(Mu)

    for k in range(K):
        sum_up = 0
        sum_down = 0
        for n in range(N):
            sum_up += R[n,k]*X[n]
            sum_down += R[n,k]
        if sum_down == 0:
            sum_down = Mu[k]    
        new_mu[k] = sum_up / sum_down
    
    return new_mu


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_standard, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    Js = []
    for i in range(num_its):
        dist = distance_matrix(X_standard,Mu)
        R = determine_r(dist)
        J = determine_j(R,dist)
        Js.append(J)
        Mu = update_Mu(Mu,X_standard,R)
    
    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js


def _plot_j():
    X, y, c = load_iris()
    a=k_means(X, 4, 10)
    b = a[2]
    plt.plot(b)
    plt.show()


def _plot_multi_j():
    ...


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> Union[float, np.ndarray]:

    Mu, R, Js= k_means(X,len(classes),num_its)
    N = R.shape[0]
    prediction = np.zeros(N)
    
    for i in range(N):
        itemindex = np.where(R[i]==1)[0][0]
        prediction[i] = itemindex
    prediction = prediction.astype(int)
    divide = int((N)/len(classes))
    value =[]
    for i in range(len(classes)):
        value.append(np.bincount(prediction[i*divide:divide+divide*i]).argmax())
    value = np.array(value)
    out = []
    for i in range(N):
        # here we locate what index this is.
        result = prediction[i]
        itemindex = np.where(value==result)[0][0]
        out.append(classes[itemindex])
    
    return out


def _iris_kmeans_accuracy():
    ...


def _my_kmeans_on_image():
    ...


def plot_image_clusters(n_clusters: int):

    image, (w, h) = image_to_numpy()
    ...
    plt.subplot('121')
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot('122')
    # uncomment the following line to run
    # plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


def _gmm_info():
    ...


def _plot_gmm():
    ...


