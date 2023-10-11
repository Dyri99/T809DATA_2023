#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Author: Dýrmundur Helgi R. Óskarsson
# Date:
# Project: 07 K-means Clustering
# Acknowledgements: Torfi & Einar
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


# Section 1.1

def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    out = np.zeros((X.shape[0], Mu.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Mu.shape[0]):
            out[i, j] = np.linalg.norm(X[i] - Mu[j])
    return out


# Section 1.2

def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    r = np.zeros(dist.shape, dtype=int)

    min_indices = np.argmin(dist, axis=1)
    for i, index in enumerate(min_indices):
        r[i, index] = 1
    return r


# Section 1.3

def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    return np.sum(R * dist) / R.shape[0]


# Section 1.4

def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    num = np.dot(R.T, X)
    den = R.sum(axis=0)[:, np.newaxis]
    
    return num / den


# Section 1.5

def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    Js = []
    for _ in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Mu = update_Mu(Mu, X_standard, R)
        Js.append(determine_j(R, dist))

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, Js


# Section 1.6
def _plot_j():
    plt.plot(Js)
    

# Section 1.7

def _plot_multi_j(ks):
    for k in ks:
        _, _, Js = k_means(X, k, 10)
        plt.plot(Js)
    plt.show()


# Section 1.9

def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    k=len(classes)
    _, R, _ = k_means(X, k, num_its)
    
    cluster_labels = {}
    for i in range(k):
        indices = np.where(R[:, i]==1)[0]
        most_common_class = np.bincount(t[indices]).argmax()
        cluster_labels[i] = most_common_class
        
    predictions = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        cluster_idx = np.where(R[i]==1)[0][0]
        predictions[i] = cluster_labels[cluster_idx]
    
    return predictions


# Section 1.10

def iris_kmeans_accuracy(true_labels: np.ndarray, kmeans_predictions: np.ndarray) -> tuple:
    accuracy = accuracy_score(true_labels, kmeans_predictions)
    conf_matrix = confusion_matrix(true_labels, kmeans_predictions)
    
    return accuracy, conf_matrix

#X, y, c = load_iris()
#classes = [0, 1, 2]
#kmeans_predictions = k_means_predict(X, y, classes, 5)
#accuracy, conf_matrix = iris_kmeans_accuracy(y, kmeans_predictions)

#print("Accuracy:", accuracy)
#print("Confusion Matrix:\n", conf_matrix)


# Section 2.1

def _my_kmeans_on_image():
    pass


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image)
    
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.title(f'Clusters: {n_clusters}')
    plt.show()
    
#for clusters in [2, 5, 10, 20]:
#    plot_image_clusters(clusters)




