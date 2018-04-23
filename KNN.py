"""
Author      : Madison Hobbs
Class       : HMC CS 158
Date        : 2018 April 14
Description : Random Forest Tuning and Performance
"""

import util
import numpy as np
import matplotlib.pyplot as plt
import validations
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn import RandomForestClassifier

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

    rf = RandomForestClassifier(class_weight=class_weight, criterion="entropy")
    # automatically does stratified kfold
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1, scoring=scoring)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_, rf_random.best_score_