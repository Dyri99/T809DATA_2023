# Author: Hnikarr Bjarmi Franklínsson
# Date: 09.10.2022
# Project: Project 8
# Acknowledgements: got a lot of help from Almar Geir Alfreðsson and just mostly copied his functions
#



from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt

#Section 1

#dæmi 1.1


def _plot_linear_kernel():
    X, t = make_blobs(40, centers=2)
    clf = svm.SVC(C=1000,kernel='linear')
    clf.fit(X,t)
    plot_svm_margin(clf, X, t,)



#dæmi 1.2

#see pdf

#dæmi 1.3

def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):

    plt.subplot(1, num_plots, index)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=30,cmap=plt.cm.Paired)
   
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')

def _compare_gamma():
    st1 = 'There are: {} support vectors in total for gamma value = {}'
    st2 = '{} belong to class blue , and {} belong to class red'
    st3 = 'The shape of the decision boundary is {}'

    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    bound = 'linear'
    g = 'default'
    clf = svm.SVC(C=1000,kernel='rbf')
    clf.fit(X,t)
    print(st1.format(clf.support_.shape[0],g))
    print(st2.format(clf.n_support_[0],clf.n_support_[1]))
    print(st3.format(bound))
    print()
   
    _subplot_svm_margin(clf, X, t, 3, 1)

    bound = 'non-linear'
    g = 0.2
    clf = svm.SVC(C=1000,kernel='rbf',gamma=g)
    clf.fit(X,t)
    print(st1.format(clf.support_.shape[0],g))
    print(st2.format(clf.n_support_[0],clf.n_support_[1]))
    print(st3.format(bound))
    print()
   
    _subplot_svm_margin(clf, X, t, 3, 2)

    bound = 'non-linear'
    g = 2
    clf = svm.SVC(C=1000,kernel='rbf',gamma=g)
    clf.fit(X,t)
    print(st1.format(clf.support_.shape[0],g))
    print(st2.format(clf.n_support_[0],clf.n_support_[1]))
    print(st3.format(bound))
    print()
   
    _subplot_svm_margin(clf, X, t, 3, 3)
    plt.show()


#dæmi 1.4

#see pdf

#dæmi 1.5

def _compare_C():
    st1 = 'There are: {} support vectors in total for C value = {}'
    st2 = '{} belong to class blue , and {} belong to class red'
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
    C = [1000,0.5,0.3,0.05,0.0001]
    for i in range(len(C)):
        clf = svm.SVC(C=C[i],kernel='linear')
        clf.fit(X,t)
        _subplot_svm_margin(clf, X, t, len(C), i+1)
        print(st1.format(clf.support_.shape[0],C[i]))
        print(st2.format(clf.n_support_[0],clf.n_support_[1]))
        plt.xlabel('Feature 1')
        if C[i]==1000:
            plt.ylabel('Feature 2')
        print()
    plt.show()


#dæmi 1.6

#see pdf


#Section 2

#dæmi 2.1

def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):

    svc.fit(X_train,t_train)
    predicted = svc.predict(X_test)
    accuracy = accuracy_score(t_test,predicted)
    precision = precision_score(t_test,predicted)
    recall = recall_score(t_test,predicted)

    return accuracy, precision, recall

#dæmi 2.2

#see pdf