"""
Author      : Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 18
Description : Perceptron Tuning and Performance
"""

import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron



def tune(X_train, y_train, scoring):

    # penalty: Regularization Term -> 'l2' or 'l1' or 'elasticnet'
    penalty = [None, 'l2', 'l1', 'elasticnet']
    # alpha: (float) constant that multiplies regulatrization term
    alpha = [float(10.0**x) for x in np.linspace(-3, 3, num = 6+1)]
    # max_iter: (int)
    max_iter = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Create the random grid
    random_grid = {'penalty': penalty,
                    'alpha': alpha,
                    'max_iter': max_iter}

    fracNeg = len(y_train[y_train == 0])/float(len(y_train))
    weight = (1-fracNeg)/float(fracNeg) 
    class_weight = {1:1, 0:weight}

    perceptron = Perceptron(class_weight=class_weight)

    perceptron_random = \
        RandomizedSearchCV(estimator = perceptron, param_distributions = random_grid, \
                            n_iter = 100, cv = 3, verbose=2, random_state=42, \
                            n_jobs = -1, scoring=scoring)
    
    perceptron_random.fit(X_train, y_train)
    return perceptron_random.best_params_



print "Getting training and test set"
#X_train, X_test, y_train, y_test = util.make_test_train()
X_train = pd.read_csv("X_train.csv")
colnames = list(X_train)
X_train = X_train.values
print X_train.shape
X_test = pd.read_csv("X_test.csv")
X_test = X_test.values
print X_test.shape
y_train = pd.read_csv("y_train.csv")['0']
y_train = y_train.values
print y_train.shape
y_test = pd.read_csv("y_test.csv")['0']
y_test =  y_test.values
print y_test.shape

colnames = list(X_train)

fracNeg = len(y_train[y_train == 0])/float(len(y_train))
print "fraction of data training examples which have negative response: " + str(fracNeg)
fracPos = len(y_train[y_train == 1])/float(len(y_train))
print "fraction of data training examples which have positive response: " + str(fracPos)


# minority class gets larger weight (as per example in class)
weight = (1-fracNeg)/float(fracNeg) 
print "negative weight is: " + str(weight)
class_weight = {1:1, 0:weight}



# Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, \
#             max_iter=None, tol=None, shuffle=True, \
#             verbose=0, eta0=1.0, n_jobs=1, random_state=0, \
#             class_weight=None, warm_start=False, n_iter=None)



#print tune(X_train, y_train, 'f1_weighted')
# 3 fold best for f1_weighted is {'penalty': 'l1', 'alpha': 0.001, 'max_iter': 1000}

perceptron_f1 = Perceptron(penalty='l1',alpha=0.001,max_iter=1000,class_weight=class_weight,random_state=42)
perceptron_f1.fit(X_train, y_train)
preds = perceptron_f1.predict(X_train)
print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) # 0.86483
preds = perceptron_f1.predict(X_test)
print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) # 0.81388
print "confusion matrix trained with f1: \n" + str(confusion_matrix(y_test, preds))
 # [[ 21   8]
 #  [ 35 140]]



#print tune(X_train, y_train, 'accuracy')
# 3 fold best for accuracy is {'penalty': 'l2', 'alpha': 1000.0, 'max_iter': 1600}

perceptron_accuracy = Perceptron(penalty='l2',alpha=1000.0,max_iter=1600,class_weight=class_weight,random_state=42)
perceptron_accuracy.fit(X_train, y_train)
preds = perceptron_accuracy.predict(X_train)
print "train accuracy_score: " + str(accuracy_score(y_train, preds)) # 0.32038
preds = perceptron_accuracy.predict(X_test)
print "test accuracy_score: " + str(accuracy_score(y_test, preds)) # 0.3186
print "confusion matrix trained with accuracy: \n" + str(confusion_matrix(y_test, preds))
# [[ 28   1]
#  [138  37]]



#print tune(X_train, y_train, 'average_precision')
# 3 fold best for average_precision is {'penalty': 'l1', 'alpha': 0.01, 'max_iter': 2000}

# perceptron_precision = Perceptron(penalty='l1',alpha=0.01,max_iter=2000,class_weight=class_weight,random_state=42)
# perceptron_precision.fit(X_train, y_train)
# preds = perceptron_precision.predict(X_train)
# print "train precision: " + str(average_precision_score(y_train, preds)) # 0.944773065198
# preds = perceptron_precision.predict(X_test)
# print "test precision: " + str(average_precision_score(y_test, preds)) # 0.918795518207
# print "confusion matrix trained with precision: \n" + str(confusion_matrix(y_test, preds))
# # [[ 21   8]
# #  [ 47 128]]




