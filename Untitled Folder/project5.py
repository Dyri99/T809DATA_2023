# Author: Hnikarr Bjarmi Franklínsson
# Date: 16. 09. 2022
# Project: Project 5
# Acknowledgements:

from typing import Union
import numpy as np

from tools import load_iris, split_train_test

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)

#Section 1

#dæmi 1.1

def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x >= -100:
        sigma = 1 / (1+np.exp(-x))
    else:
        sigma = 0.0
    
    return sigma


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    sigma = sigmoid(x) * (1-sigmoid(x))
    
    return sigma

#a = sigmoid(-0.78715749)
#b = d_sigmoid(0.2)
#print(a)
#print(b)


#dæmi 1.2

def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    prod = x * w
    sumproduct = sum(prod)
    sigma = sigmoid(sumproduct)

    return np.array([sumproduct, sigma])

#a,z = perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))
#print(a)
#print(z)
#b = perceptron(np.array([0.2,0.4]),np.array([0.1,0.4]))
#print(b)


#dæmi 1.3

def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    z0 = np.insert(x,0,1)
    a1 = np.zeros(M, dtype=float)
    z1 = np.zeros(M, dtype=float)

    for i in range(M):
        a1[i], z1[i] = perceptron(z0, W1[:,i])

    z1 = np.insert(z1,0,1)
    a2 = np.zeros(K, dtype=float)
    y = np.zeros(K, dtype=float)

    for i in range(K):
        a2[i], y[i] = perceptron(z1, W2[:,i])

    return y, z0, z1, a1, a2
    


""" np.random.seed(1234)

x = train_features[0, :]
K = 3 # number of classes
M = 10
D = 4
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1


y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
print(y)
print(z0)
print(z1)
print(a1)
print(a2) """

    