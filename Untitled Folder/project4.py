# Author: Hnikarr Bjarmi Franklínsson
# Date: 11. 09. 2022
# Project: Project 4
# Acknowledgements: got some help from Oliver Aron Jóhannesson with the maximum_likelihood function, I did it mostly by myself, but it wasn't quite right
# got much help from Oliver Aron Jóhannesson with the predict function, I had no idea what to do until I saw his function, and then I made my own
# I just copied the maximum_aposteriori from Oliver Aron Jóhannesson, I had no idea what to do, and I didn't understand what he was doing when he showed me his code so I couldn't even make it my own. I copied it so I could at least try to get section 2.2


from statistics import mean
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)


#Section 1

#dæmi 1.1

def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:

    len1 = len(features)
    len2 = features.shape[1]
    len3 = len(train_targets[train_targets==selected_class])
    array = np.zeros((len3,len2), dtype=float)
    mean_array = []
    b = 0
    for i in range(len1):
        if targets[i] == selected_class:
            for x in range(len2):
                array [b,x] = features[i,x]
            b = b + 1            
    for j in range(len2):
        counter = 0
        for z in range(len3):
            counter = counter + array[z,j]
        mean = counter/len3
        mean_array.append(mean)
    
    return mean_array


#a = mean_of_class(train_features, train_targets, 0)
#print(a)

#dæmi 1.2

def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:

    len1 = len(features)
    len2 = features.shape[1]
    len3 = len(train_targets[train_targets==0])
    array = []
    for i in range(len1):
        if targets[i] == selected_class:
            array.append(features[i])  
    cov = np.cov(array, rowvar=False)


    return cov

#a = covar_of_class(train_features, train_targets, 0)
#print(a)

#dæmi 1.3

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:

    prob = multivariate_normal(mean = class_mean, cov = class_covar).pdf(feature)

    return prob

""" class_mean = mean_of_class(train_features, train_targets, 0)
class_cov = covar_of_class(train_features, train_targets, 0)
print(class_mean)
print(class_cov)
a = likelihood_of_class(test_features[0, :], class_mean, class_cov)
print(a)
 """
#dæmi 1.4

def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:

    means, covs = [], []
    for class_label in classes:
        class_mean = mean_of_class(train_features, train_targets, class_label)
        means.append(class_mean)
        class_cov = covar_of_class(train_features, train_targets, class_label)
        covs.append(class_cov)
    likelihoods = []
    for i in range(test_features.shape[0]):
        prob_each_class = []
        for j in range(len(classes)):
            class_likelihood = likelihood_of_class(test_features[i], means[j], covs[j])
            prob_each_class.append(class_likelihood)
        likelihoods.append(prob_each_class)
    return np.array(likelihoods)

#a = maximum_likelihood(train_features, train_targets, test_features, classes)
#print(a)

#dæmi 1.5


def predict(likelihoods: np.ndarray):

    class_array = np.zeros(likelihoods.shape[0])
    for i in range(likelihoods.shape[0]):
        class_array [i] = np.argmax(likelihoods[i])
    
    return class_array




likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
#print(likelihoods)
#prediction = predict(likelihoods)
#print(prediction)


""" 
Ok I'm going to be real with you, I had no idea how the functions
in help.py work and what they do, so it was hard for me to know
what I was doing from section 1.2 to 1.5, but somehow I managed
to do it somewhat right by just putting something in despite having
very almost no clue what I was actually doing. I could do 1.2 and 1.3
by my self but needed help with 1.4 and 1.5 (see Acknowledgements) 
"""


#Section 2

#dæmi 2.1
import pandas as pd

def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    framet = pd.DataFrame(train_features)
    framet[4] = train_targets
    
    dict = {}
    for i in classes:
        dict[i] = framet[framet[4]==i].shape[0] / framet.shape[0]
    frame_like = pd.DataFrame(maximum_likelihood(train_features,train_targets,test_features,classes))
    for i in dict:
        frame_like[i] = frame_like[i] * dict[i]
    frame_like = np.array(frame_like)
    return frame_like



#section 2.2
def confusion_matrix(
    input: np.ndarray,
    test_targets: np.ndarray,
    classes: list
) -> np.ndarray:
    predicted = predict(input)
    actual = test_targets 
    length = len(classes)
    conmatr = np.zeros((length,length), dtype=int)
    for i in range(length):
        for j in range(length):
            conmatr [i,j] = np.sum(((actual == i) & (predicted == j)))
    return conmatr





""" likelihoods1 = maximum_aposteriori(train_features, train_targets, test_features, classes)
a = confusion_matrix(likelihoods1, test_targets, classes)
print(a)
likelihoods2 = maximum_likelihood(train_features, train_targets, test_features, classes)
b = confusion_matrix(likelihoods2, test_targets, classes)
print(b) """