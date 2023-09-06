#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Author: Dýrmundur Helgi R. Óskarsson
# Date: 30.8.2023
# Project: Assignment 2 - K nearest neighbours
# Acknowledgements: template.prior() from project 1 - decision trees
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

#d, t, classes = load_iris()
#plot_points(d, t)


# In[14]:


# Part 1.1
def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return (np.sqrt(np.sum((x - y)**2)))

#d, t, classes = load_iris()
#x, points = d[0,:], d[1:, :]
#x_target, point_targets = t[0], t[1:]
#print(euclidian_distance(x, points[0]))
#print(euclidian_distance(x, points[50]))


# In[15]:


# Part 1.2

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and many
    points
    '''
    distances = np.zeros(points.shape[0]) # np.shape[0] gives us # of rows
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances

#euclidian_distances(x, points)


# In[16]:


# Part 1.3

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    dist = euclidian_distances(x, points)
    nearest = np.argsort(dist)[:k]
    return nearest

#print(k_nearest(x, points, 1))
#print(k_nearest(x, points, 3))


# In[17]:


# Part 1.4    

def vote(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    probabilities = np.zeros(len(classes)) # use len() because it is a list
    
    for i, a_class in enumerate(classes):
        for t in targets:
            if t == a_class:
                probabilities[i] += 1
    return classes[np.argmax(probabilities)]

#print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
#print(vote(np.array([1,1,1,1]), np.array([0,1])))


# In[18]:


# Part 1.5

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    k_closest = k_nearest(x, points, k)
    closest_targets = point_targets[k_closest]
    return vote(closest_targets, classes)

#print(knn(x, points, point_targets, classes, 1))
#print(knn(x, points, point_targets, classes, 5))
#print(knn(x, points, point_targets, classes, 150))


# In[19]:


# Part 2
import help

#d, t, classes = load_iris()
#(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)


# In[20]:


# Part 2.1

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    prediction = []
    
    for i, x in enumerate(points):
        point = help.remove_one(points, i)
        point_target = help.remove_one(point_targets, i)
        prediction.append(knn(x, point, point_target, classes, k))
    
    return np.array(prediction)

#print(knn_predict(d_test, t_test, classes, 10))
#print(knn_predict(d_test, t_test, classes, 5))


# In[21]:


# Part 2.2

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    prediction = knn_predict(points, point_targets, classes, k)
    correct_predictions = np.sum(prediction == point_targets)
    accuracy = correct_predictions / points.shape[0]
    
    return accuracy

#print(knn_accuracy(d_test, t_test, classes, 10))
#print(knn_accuracy(d_test, t_test, classes, 5))


# In[22]:


# Part 2.3

def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    
    prediction = knn_predict(points, point_targets, classes, k)
    length = len(classes)
    confusion_matrix = np.zeros((length, length), dtype='int')
    
    for i in range(len(point_targets)):
        correct = point_targets[i]
        guess = prediction[i]
        confusion_matrix[guess][correct] += 1
    return confusion_matrix

#print(knn_confusion_matrix(d_test, t_test, classes, 10))
#print(knn_confusion_matrix(d_test, t_test, classes, 20))


# In[23]:


# Part 2.4

def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list
) -> int:
    
    top_k = None
    highest_acc = 0
    for k in range(1, points.shape[0]): #[1, N-1]
        acc = knn_accuracy(points, point_targets, classes, k)
        if acc > highest_acc:
            top_k = k
            highest_acc = acc
    #print(highest_acc)
    return top_k

#best_k(d_train, t_train, classes)


# In[24]:


# Part 2.5

def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ['yellow', 'purple', 'blue']
    
    prediction = knn_predict(points, point_targets, classes, k)
    correct = (prediction == point_targets)
    
    plt.figure(figsize=(8,6))
    for i in range(points.shape[0]):
        [x, y] = points[i, :2]
        edgecolor = 'green' if correct[i] else 'red'
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors=edgecolor, linewidths=2)
    
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.savefig("2_5_1.png")
    plt.show()

#knn_plot_points(d, t, classes, 3)


# In[ ]:




