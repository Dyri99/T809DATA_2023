#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Dýrmundur Helgi
# Date: 26.09.2023
# Project: 
# Acknowledgements: Einar & Torfi
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


# In[2]:


# Section 1.1

def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

#print(standardize(np.array([[0, 0], [0, 0], [1, 1], [1, 1]])))


# In[3]:


# Section 1.2

def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    standardized_X = standardize(X)
    
    x_dim = standardized_X[:, i]
    y_dim = standardized_X[:, j]
    
    plt.scatter(x_dim, y_dim)
    plt.xlabel(f'Dimension {i}')
    plt.ylabel(f'Dimension {j}')
    #plt.title('Scatter plot of standardized dimensions')
    
#X = np.array([
#    [1, 2, 3, 4],
#    [0, 0, 0, 0],
#    [4, 5, 5, 4],
#    [2, 2, 2, 2],
#    [8, 6, 4, 2]])
#scatter_standardized_dims(X, 0, 2)


# In[4]:


# Section 1.3

def _scatter_cancer():
    X, y = load_cancer()
    standardized_X = standardize(X[:5])
    plt.figure(figsize=(15, 10))
    for i in range(X.shape[1]):
        plt.subplot(5,6,i+1)
        scatter_standardized_dims(X[:5], 0, i)
    plt.tight_layout()
    
#_scatter_cancer()


# In[16]:


# Section 2.1

def _plot_pca_components():
    X, y = load_cancer()
    standardized_X = standardize(X)
    pca = PCA(n_components = X.shape[1])
    principal_components = pca.fit_transform(standardized_X)
    plt.figure(figsize=(20,16))
    for i in range(3):
        plt.subplot(5, 6, i+1)
        plt.plot(range(len(principal_components)), principal_components[:, i])
        plt.title(f'Component {i}')


#_plot_pca_components()


# In[24]:


# Section 3.1

def _plot_eigen_values():
    X, y = load_cancer()
    standardized_X = standardize(X)
    
    pca = PCA(n_components = min(X.shape))
    pca.fit(standardized_X)
    
    eigen_values = pca.explained_variance_
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigen_values) + 1), eigen_values, marker='o')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

#_plot_eigen_values()


# In[25]:


# Section 3.2

def _plot_log_eigen_values():
    X, y = load_cancer()
    standardized_X = standardize(X)
    
    pca = PCA(n_components = min(X.shape))
    pca.fit(standardized_X)
    
    eigen_values = pca.explained_variance_
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigen_values) + 1), np.log10(eigen_values), marker='o')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()
    
#_plot_log_eigen_values()


# In[26]:


# Section 3.3

def _plot_cum_variance():
    X, y = load_cancer()
    standardized_X = standardize(X)
    
    pca = PCA(n_components = min(X.shape))
    pca.fit(standardized_X)
    
    eigen_values = pca.explained_variance_
    cumulative_variance = np.cumsum(eigen_values) / np.sum(eigen_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()
    
#_plot_cum_variance()


# In[ ]:




