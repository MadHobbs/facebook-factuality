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

    rf = RandomForestClassifier(class_weight=class_weight, criterion="entropy")
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1, scoring=scoring)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_
    

def main():
    X_train, X_test, y_train, y_test, colnames = util.make_test_train()

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
    
    # {'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 1000, 'max_features': 'auto', 'min_samples_split': 22, 'max_depth': 110}

    # 5 fold best trained on entropy ones {'bootstrap': False, 'min_samples_leaf': 2, 'n_estimators': 1600, 'max_features': 'sqrt', 'min_samples_split': 27, 'max_depth': 80}
    # smaller search area : {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 400, 'max_features': 'sqrt', 'min_samples_split': 5, 'max_depth': 100}
    rf_f1 = RandomForestClassifier(class_weight=class_weight, bootstrap = True, min_samples_leaf = 1, n_estimators =  400, max_features = 'sqrt', min_samples_split = 5, max_depth = 100, criterion="entropy")
    rf_f1.fit(X_train, y_train)
    preds = rf_f1.predict(X_train)
    print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) # 0.9958
    preds = rf_f1.predict(X_test)
    print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) # 0.88877
    print "confusion matrix trained with f1: \n" + str(confusion_matrix(y_test, preds))

    importances = rf_f1.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_f1.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, colnames[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    #plt.figure()
    #plt.title("Feature importances")
    #plt.bar(range(X_train.shape[1]), importances[indices],
       #color="r", yerr=std[indices], align="center")
    #plt.xticks(range(X_train.shape[1]), indices)
    #plt.xlim([-1, X_train.shape[1]])
    #plt.show()

    #validations.check_overfit(rf_f1, f1_score, "weighted")

    # print tune(X_train, y_train, 'accuracy')
    # 3 fold best for accuracy: best is {'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 200, 'max_features': 'sqrt', 'min_samples_split': 2, 'max_depth': 50}
    #rf_accuracy = RandomForestClassifier(class_weight=class_weight, bootstrap = True, min_samples_leaf = 1, n_estimators = 200, max_features = 'sqrt', min_samples_split = 2, max_depth = 50, criterion="entropy")
    rf_accuracy = rf_f1
    rf_accuracy.fit(X_train, y_train)
    preds = rf_accuracy.predict(X_train)
    print "train accuracy: " + str(accuracy_score(y_train, preds)) # 1.0
    preds = rf_accuracy.predict(X_test)
    print "test accuracy: " + str(accuracy_score(y_test, preds))
    print "confusion matrix trained with accuracy: \n" + str(confusion_matrix(y_test, preds)) #0.887
    
    #validations.check_overfit(rf_accuracy, accuracy_score)

    #print tune(X_train, y_train, 'average_precision')
    # 3 fold best for precision is {'bootstrap': False, 'min_samples_leaf': 1, 'n_estimators': 800, 'max_features': 'sqrt', 'min_samples_split': 5, 'max_depth': 90}
    #rf_precision = RandomForestClassifier(class_weight=class_weight, bootstrap = False, min_samples_leaf = 1, n_estimators = 800, max_features = 'sqrt', min_samples_split = 5, max_depth = 90, criterion="entropy")
    rf_precision = rf_f1
    rf_precision.fit(X_train, y_train)
    preds = rf_precision.predict(X_train)
    print "train precision: " + str(average_precision_score(y_train, preds)) # 0.9999
    preds = rf_precision.predict(X_test)
    print "test precision: " + str(average_precision_score(y_test, preds)) # 0.96h
    print "confusion matrix trained with precision: \n" + str(confusion_matrix(y_test, preds))