"""
Author      : Madison Hobbs
Class       : HMC CS 158
Date        : 2018 April 14
Description : Random Forest Tuning and Performance
"""

import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune(X_train, y_train, scoring):
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    fracNeg = len(y_train[y_train == 0])/float(len(y_train))
    weight = (1-fracNeg)/float(fracNeg) 
    class_weight = {1:1, 0:weight}

    rf = RandomForestClassifier(class_weight=class_weight)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=scoring)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_

def main():
    X_train, X_test, y_train, y_test = util.make_test_train()

    fracNeg = len(y_train[y_train == 0])/float(len(y_train))
    print "fraction of data training examples which have negative response: " + str(fracNeg)
    fracPos = len(y_train[y_train == 1])/float(len(y_train))
    print "fraction of data training examples which have positive response: " + str(fracPos)
    # minority class gets larger weight (as per example in class)
    weight = (1-fracNeg)/float(fracNeg) 
    print "negative weight is: " + str(weight)
    class_weight = {1:1, 0:weight}

    # print tune(X_train, y_train, 'f1_weighted')
    # 3 fold best for f1_weighted = {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 400, 
    # 'max_features': 'sqrt', 'min_samples_split': 5, 'max_depth': 30}
    rf_f1 = RandomForestClassifier(class_weight=class_weight, bootstrap = True, min_samples_leaf = 1, n_estimators =  400, max_features = 'sqrt', min_samples_split = 5, max_depth = 30)
    rf_f1.fit(X_train, y_train)
    preds = rf_f1.predict(X_train)
    print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) # 0.9958
    preds = rf_f1.predict(X_test)
    print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) # 0.88877

    #print tune(X_train, y_train, 'accuracy')
    # 3 fold best for accuracy: best is {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 400, 'max_features': 'sqrt', 'min_samples_split': 5, 'max_depth': 30}
    #rf_accuracy = RandomForestClassifier(class_weight=class_weight, bootstrap = True, min_samples_leaf = 1, n_estimators =  400, max_features = 'sqrt', min_samples_split = 5, max_depth = 30)
    #rf_accuracy.fit(X_train, y_train)
    #preds = rf_accuracy.predict(X_train)
    #print "train accuracy: " + str(accuracy_score(y_train, preds, average="weighted"))
    #preds = rf_accuracy.predict(X_test)
    #print "test accuracy: " + str(accuracy_score(y_test, preds, average="weighted"))
    
    print tune(X_train, y_train, 'average_precision')

    # accuracy on left, precision on right