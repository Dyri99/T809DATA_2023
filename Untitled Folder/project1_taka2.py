# Author: Hnikarr Bjarmi Franklínsson
# Date: 24. 08. 2022
# Project: 1
# Acknowledgements: prior function taken from lecture
# split_data function mostly taken from lecture
# IrisTreeTrainer.train(self) taken from lecture
# got IrisTreeTrainer.accuracy(self) from Baldvin Egill Baldvinsson
# worked on IrisTreeTrainer.confusion_matrix(self) with Baldvin Egill Baldvinsson


from operator import length_hint, truediv
from pickle import TRUE
from tkinter import N
from tkinter.tix import Tree
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree


from tools1 import load_iris, split_train_test
features, targets, classes = load_iris()

#Part 1

#dæmi 1.1

def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    sample_count = len(targets)
    class_probs = []
    for c in classes:
        class_count = 0
        for t in targets:
            if t == c:
                class_count += 1
        class_probs.append(class_count / sample_count)
    
    #print(class_probs)
    return class_probs

#print(prior(np.array([0,0,1,2]), np.array([0,1,2])))

#dæmi1.2

def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    filter1 = features[:,split_feature_index] < theta
    filter2 = features[:,split_feature_index] >= theta

    features_1 = features[filter1,:]
    targets_1 = targets[filter1]
    num_feat1 = list(filter1).count(1)

    features_2 = features[filter2,:]
    targets_2 = targets[filter2]
    num_feat2 = list(filter2).count(1)
    

    #print('f_1 should contain', num_feat1, 'and f_2 contain', num_feat2, 'samples.')
    return (features_1, targets_1), (features_2, targets_2)


#(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)

#Dæmi 1.3

def gini_impurity(targets: np.ndarray, classes: list) -> float:    
    class_probs = prior(targets, classes)
    subtr = 1
    for c in classes:
        subtr = subtr - np.square(class_probs[c])
    gini_imp = 0.5 * subtr
    #print(gini_imp)
    return gini_imp




#Dæmi 1.4

def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n1 = t1.shape[0]
    n2 = t2.shape[0]
    n = n1 + n2

    weight_imp = (g1*n1+g2*n2)/n

    #print(weight_imp)
    return weight_imp


#Dæmi 1.5

def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    weight_imp = weighted_impurity(t_1, t_2, classes)

    #print(weight_imp)
    return weight_imp

#total_gini_impurity(features, targets, classes, 2, 4.65)

#Dæmi 1.6

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    num_points = num_tries
    best_gini = 10000
    best_dim = 0
    best_theta = 0
    
    #best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        values = np.array(features[:,i])
        min_value = values.min()
        max_value = values.max()
        # create the thresholds
        thetas = np.linspace(min_value, max_value, num_points+2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            weight_imp = total_gini_impurity(features, targets, classes, i, theta)
            if weight_imp < best_gini:
                best_gini = weight_imp
                best_dim = i 
                best_theta = theta

    #print(best_gini, best_dim, best_theta)
    return best_gini, best_dim, best_theta

#a = brute_best_split(features, targets, classes, 30)
#print(a)



#Part 2

class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()
        
    #Dæmi 2.1

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)
        
    #Dæmi 2.2

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    #Dæmi 2.3

    def plot(self):
        plot_tree(self.tree)
        return plt.show()

    #Dæmi 2.4

    def guess(self):
        return(self.tree.predict(self.test_features))

    #Dæmi 2.5

    def confusion_matrix(self):
        predicted = self.guess()
        actual = self.test_targets
        length = len(self.classes)
        conmatr = np.zeros((length,length), dtype=int)
        for i in range(length):
            for j in range(length):
                conmatr [i,j] = np.sum(((actual == i) & (predicted == j)))
        return conmatr


""" features, targets, classes = load_iris()
dt = IrisTreeTrainer(features, targets, classes=classes)
dt.train()
print(f'The accuracy is: {dt.accuracy()}')
dt.plot()
print(f'I guessed: {dt.guess()}')
print(f'The true targets are: {dt.test_targets}')
print(dt.confusion_matrix()) """

