#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Author: Dýrmundur Helgi R. Óskarsson
# Date: 20.9.2023
# Project: 05 Classification
# Acknowledgements: Einar Óskar & Torfi Tímóteus
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


# In[3]:


# Section 1.1

def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    class_mean = np.mean(features[targets == selected_class], axis=0)
    
    return class_mean
   

#features, targets, classes = load_iris()
#(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio = 0.6)

#print(mean_of_class(train_features, train_targets, 0))


# In[4]:


# Section 1.2

def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    class_cov = np.cov(features[targets == selected_class], rowvar=False)
    
    return class_cov

#print(covar_of_class(train_features, train_targets, 0))


# In[5]:


# Section 1.3

def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    mvn = multivariate_normal(class_mean, class_covar)
    likelihood = mvn.pdf(feature)
    
    return likelihood
    
#class_mean = mean_of_class(train_features, train_targets, 0)
#class_cov = covar_of_class(train_features, train_targets, 0)
#print(likelihood_of_class(test_features[0, :], class_mean, class_cov))


# In[6]:


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        class_mean = mean_of_class(train_features, train_targets, class_label)
        class_cov = covar_of_class(train_features, train_targets, class_label)
        
        means.append(class_mean), covs.append(class_cov)
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihood = []
        for j in range(len(classes)):
            likelihood.append(likelihood_of_class(train_features[i], means[j], covs[j]))
        likelihoods.append(likelihood)
    return np.array(likelihoods)

#print(maximum_likelihood(train_features, train_targets, test_features, classes))


# In[ ]:





# In[7]:


# Section 1.5

def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)

#likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
#print(predict(likelihoods))


# In[10]:


# Section 2.1

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
    means, covs, priors = [], [], []
    length_features = len(train_features)
    for class_label in classes:
        class_mean = mean_of_class(train_features, train_targets, class_label)
        class_cov = covar_of_class(train_features, train_targets, class_label)
        prior = len(train_features[train_targets == class_label]) / length_features
        
        means.append(class_mean), covs.append(class_cov), priors.append(prior)
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihood = []
        for j in range(len(classes)):
            likelihood.append((likelihood_of_class(train_features[i], means[j], covs[j])) * priors[j])
        likelihoods.append(likelihood)
    return np.array(likelihoods)

#posteriors = maximum_aposteriori(train_features, train_targets, test_features, classes)
#print(posteriors)

#predictions = predict(posteriors)
#print(predictions)


# In[31]:


# Section 2.2

#predictions = predict(likelihoods)
#predictions = predict(posteriors)

#print(train_targets[0:len(predictions)])
#count=0
#for i, tt in enumerate(train_targets[0:len(predictions)]):
#    if tt==predictions[i]:
#        count+=1
#print(count/len(predictions))

#length = len(classes)
#confusion_matrix = np.zeros((length, length), dtype='int')

#for i in range(len(predictions)):
#    correct = train_targets[i]
#    guess = predictions[i]
#    confusion_matrix[guess][correct] += 1
    
#print(confusion_matrix)


# In[ ]:




