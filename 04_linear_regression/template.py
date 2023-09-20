#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Author: Dýrmundur Helgi R. Óskarsson
# Date: 15.9.2023
# Project: 4 Linear Regression
# Acknowledgements: Einar Óskar og Torfi Tímóteus
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


# In[49]:


# Part 1.1

def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    N, D = features.shape
    M = mu.shape[0]
    
    fi = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            mvn = multivariate_normal(mu[j], sigma * np.eye(D))
            fi[i, j] = mvn.pdf(features[i])
    
    return fi

#X, t = load_regression_iris()
#N, D = X.shape

#M, sigma = 10, 10
#mu = np.zeros((M, D))
#for i in range(D):
#    mmin = np.min(X[i, :])
#    mmax = np.max(X[i, :])
#    mu[:, i] = np.linspace(mmin, mmax, M)
#fi = mvn_basis(X, mu, sigma)
#print(fi)


# Part 1.2

def _plot_mvn():
    for j in range(mu.shape[0]):
        plt.plot(X[:, 0], fi[:, j], label=f'Basis Function {j+1}')
    
    plt.xlabel('Feature')
    plt.ylabel('MVN Basis Function Output')
    plt.legend()
    plt.title('MVN Basis Functions')
    plt.grid(True)
    plt.savefig("plot_1_2_1.png")
    plt.show()
    
#_plot_mvn()


# In[50]:


# Part 1.3

def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    N, M = fi.shape
    I = np.eye(M)
    
    w = np.dot(np.linalg.inv(np.dot(fi.T, fi) + lamda * I), np.dot(fi.T, targets))
    
    return w

#fi = mvn_basis(X, mu, sigma)
#lamda = 0.001
#wml = max_likelihood_linreg(fi, t, lamda)
#print(wml)


# In[51]:


# Part 1.4

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    N, D = features.shape
    M = mu.shape[0]
    fi = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            mvn = multivariate_normal(mu[j], sigma * np.eye(D))
            fi[i, j] = mvn.pdf(features[i])
    
    predictions = np.dot(fi, w)
    
    return predictions

#wml = max_likelihood_linreg(fi, t, lamda)
#prediction = linear_model(X, mu, sigma, wml)
#print(prediction)


# In[52]:


#plt.plot(prediction)
#plt.plot(X)
#plt.show()


# In[ ]:





# In[ ]:




