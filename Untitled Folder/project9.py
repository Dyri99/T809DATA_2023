# Author: Hnikarr Bjarmi Franklínsson
# Date: 23. 10. 2022
# Project: Project 9
# Acknowledgements: got a lot of help from Oliver Aron Jóhannesson and just mostly copied his functions



import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score)

from collections import OrderedDict



class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train,self.t_train)

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        y_true = self.t_test
        y_pred = self.classifier.predict(self.X_test)
        return confusion_matrix(y_true,y_pred)

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        y_true = self.t_test
        y_pred = self.classifier.predict(self.X_test)
        return accuracy_score(y_true,y_pred)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        y_true = self.t_test
        y_pred = self.classifier.predict(self.X_test)
        return precision_score(y_true,y_pred)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        y_true = self.t_test
        y_pred = self.classifier.predict(self.X_test)
        return recall_score(y_true,y_pred)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return (cross_val_score(DecisionTreeClassifier(),self.X,self.t,cv=10)).mean()

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        f_importance =self.classifier.feature_importances_
        # summarize feature importance
        a_dict = {}
        output = []
        for i,v in enumerate(f_importance):
            a_dict[i] = v
        for w in sorted(a_dict, key=a_dict.get, reverse=True):
            output.append(w)
        # plot feature importance
        plt.bar([x for x in range(len(f_importance))], f_importance)
        plt.xlabel('Feature index')
        plt.ylabel('Feature importance')
        plt.savefig('2_2_1.png')
        plt.show()
        return output




def _plot_oob_error():
    cancer = load_breast_cancer()
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cancer.data,  cancer.target)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig('2_4_1.png')
    plt.show()

def _plot_extreme_oob_error():
    cancer = load_breast_cancer()
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("ExtraTreesClassifier, max_features='sqrt'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                bootstrap=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features='log2'",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE)),
        ("ExtraTreesClassifier, max_features=None",
            ExtraTreesClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(cancer.data,  cancer.target)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig('3_2_1.png')
    plt.show()