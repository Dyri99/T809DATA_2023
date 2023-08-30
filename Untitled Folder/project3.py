
# Author: Hnikarr Bjarmi Franklínsson
# Date: 02.09.2022
# Project: Project 3
# Acknowledgements: 
#

#Section 1

#dæmi 1.1

from math import sqrt
from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(1234)

def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    
    ident = np.identity(k)
    cov = (var**2) * ident

    norm_dist = np.random.multivariate_normal(mean,cov,n)
    return norm_dist



#dæmi 1.2

#sec_1_2 = gen_data(300, 3, [0, 1, -1], sqrt(3))
#print(sec_1_2)
#scatter_3d_data(sec_1_2)


#dæmi 1.3
#bar_per_axis(sec_1_2)

#dæmi 1.4
def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    upd_mean = mu + 1/n * (x - mu)
    return upd_mean



#dæmi 1.5

def _plot_sequence_estimate():
    data = gen_data(100, 3, [0, 0, 0], sqrt(3))
    len1 = data.shape[0]
    len2 = data.shape[1]
    estimates = np.zeros((1,len2), dtype = float)
    matr = np.zeros((len1,len2), dtype=float)

    for i in range(len1):
        for j in range(len2):
            matr [i,j] = estimates[0,j]
        estimates = update_sequence_mean(estimates, data[i], i+1)
        

    plt.plot([e[0] for e in matr], label='First dimension')
    plt.plot([e[1] for e in matr], label='Second dimension')
    plt.plot([e[2] for e in matr], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()

#b = _plot_sequence_estimate()

#dæmi 1.6

def _square_error():
    data = gen_data(100, 3, [0, 0, 0], sqrt(3)) #change for different values of y
    y_hat = [0, 0, 0] #change for different predictions (y_hat)
    estimates = y_hat
    len1 = data.shape[0]
    len2 = data.shape[1]
    sq_err = np.zeros((len1,1), dtype = float)
    y = np.zeros((len1,len2), dtype=float)
    for i in range(len1):
        counter = 0
        estimates = update_sequence_mean(estimates, data[i], i+1)
        for j in range(len2):
            y [i,j] = estimates[j]
            counter = counter + (y [i,j] - y_hat [j]) ** 2
        sq_err[i,0] = counter/len2
        
    return sq_err
        
        
#a = _square_error()
#print(a)

def _plot_mean_square_error():
    a = _square_error()
    plt.plot(a)
    plt.show()

#b = _plot_mean_square_error()


""" 
It didn't make any sense for me that there were only inputs in one function
and not the other so I decided to take the inputs out of the _square_error
function so I wouldn't need any inputs for the _plot_mean_square_error
function either.
If you want to test other datasets you'll have to change it inside the
function, I've marked what you have to chance regarding y and y_hat.
"""