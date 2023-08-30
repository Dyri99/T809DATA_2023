# Author:  Hnikarr Bjarmi Franklínsson
# Date: 28. 08. 2022
# Project: 2
# Acknowledgements: Got knn_predict function sent from Oliver Aron Jóhannesson just so I could go on with the assignement and atleast try for the functions after that one
# Also go knn_accuracy function sent from Oliver Aron Jóhannesson for the same resons


import numpy as np
import matplotlib.pyplot as plt

from tools2 import load_iris, split_train_test, plot_points

d, t, classes = load_iris()
x, points = d[0,:], d[1:,:]
x_target, point_targets = t[0], t[1:]
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

#Part 1


#Dæmi 1.1

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    dist = 0
    for i in range(len(x)):
        dist = dist + (x[i]-y[i]) ** 2
    dist = np.sqrt(dist)
    return dist


#Dæmi 1.2

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    
    return distances


#Dæmi 1.3

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    A = euclidian_distances(x, points)
    k = k

    idx = np.argpartition(A, k)
    lowest = idx[:k]
    return lowest


#Dæmi 1.4

def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    sample_count = len(targets)
    most_often = 0
    for c in classes:
        class_count = 0
        for t in targets:
            if t == c:
                class_count += 1
                if class_count > most_often:
                    most_often = class_count
                    num_most_often = c

    return num_most_often


#Dæmi 1.5  

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
    k_nearest_points = k_nearest(x, points, k)
    class_of_points = point_targets[k_nearest_points]
    most_common_class = vote(class_of_points, classes)
    return most_common_class

#Could not run test case for k = 150 because there are only 149 points in point_targets


#Part 2

#Dæmi 2.1
def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    knn_multiple = []

    for i in range(len(points)):
        taken_i = np.concatenate((points[0:i], points[i+1:]))
        taken_j = np.concatenate((point_targets[0:i], point_targets[i+1:]))
        knn_multiple.append(knn(points[i],taken_i,taken_j,classes,k))
    return knn_multiple


#Dæmi 2.2

from sklearn.metrics import accuracy_score
def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    knn_points = knn_predict(points,point_targets,classes,k)
    return accuracy_score(point_targets,knn_points)



#Dæmi 2.3

def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
        predicted = knn_predict(points, point_targets, classes, k)
        length = len(classes)
        conmatr = np.zeros((length,length), dtype=int)
        for i in range(length):
            for j in range(length):
                conmatr [i,j] = np.sum(((point_targets == i) & (predicted == j)))
        return conmatr
# I don't know why it doesn't run, it's the same function from project 1 for a confusion matrix
