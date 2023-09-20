#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Dýrmundur Helgi
# Date: 6.9.2023
# Project: 3 Sequential Estimation
# Acknowledgements: Einar Óskar & Torfi Tímóteus
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# Part 1.1

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    # np.eye(k) returns I_k
    cov_matrix = (var**2) * np.eye(k)
    
    X = np.random.multivariate_normal(mean, cov_matrix, n)
    
    return X

#np.random.seed(1234)
#print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))

#np.random.seed(1234)
#print(gen_data(5, 1, np.array([0.5]), 0.5))


# In[3]:


# Part 1.2

# Parameters
n = 300; k = 3
mean = [0, 1, -1]
variance = np.sqrt(3)

np.random.seed(1234)
data = gen_data(n, k, mean, variance)

#scatter_3d_data(data)
#bar_per_axis(data)

#file_path = '1_2_1.txt'
#np.savetxt(file_path, data, fmt='%.6f')


# In[4]:


# Part 1.4

def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    
    return mu + (1/n)*(x-mu)

#X = data[-1]
#np.random.seed(1234)
#mean = np.mean(X, 0); print(mean)
#new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
#update_sequence_mean(mean, new_x, X.shape[0])


# In[5]:


# Part 1.5

def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), 4)
    estimates = np.zeros((101,3))
    for i in range(data.shape[0]):
        estimates[i+1] = update_sequence_mean(estimates[i], data[i], i+1)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.savefig("1_5_1.png")
    plt.show()

#np.random.seed(1234)
#_plot_sequence_estimate()


# In[6]:


# Part 1.6

def _square_error(y, y_hat):
    return (y - y_hat)**2


def _plot_mean_square_error(square_error):
    plt.plot(np.mean(square_error, 1))
    plt.savefig("1_6_1.png")

#np.random.seed(1234)
#data = gen_data(100, 3, np.array([0, 0, 0]), 4)

#estimates = np.zeros((101,3))

#for i in range(data.shape[0]):
#    estimates[i+1] = update_sequence_mean(estimates[i], data[i], i+1)
    
#square_error = _square_error(estimates[1:], [0,0,0])

#_plot_mean_square_error(square_error)


# In[ ]:




