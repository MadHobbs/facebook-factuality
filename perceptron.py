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

import pandas as pd



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
X_train = pd.read_csv("X_train.csv").drop(['Unnamed: 0'], axis = 1)
print X_train.columns
X_train = X_train.values
print X_train.shape
X_test = pd.read_csv("X_test.csv").drop(['Unnamed: 0'], axis = 1)
print X_test.columns
X_test = X_test.values
print X_test.shape
y_train = pd.read_csv("y_train.csv")['0']
y_train = y_train.values
print y_train.shape
y_test = pd.read_csv("y_test.csv")['0']
y_test =  y_test.values
print y_test.shape

colnames = [u'2016', u'actual', u'address', u'aid', u'alleg', u'also', u'america', u'american', u'among', u'back', u'bad', u'barack', u'battleground', u'berni', u'best', u'better', u'bill', u'black', u'board', u'break', u'bush', u'busi', u'bust', u'call', u'came', u'campaign', u'candid', u'carolina', u'caught', u'chariti', u'check', u'children', u'chris', u'christi', u'citi', u'claim', u'clinton', u'cnn', u'come', u'communiti', u'compani', u'controversi', u'countri', u'court', u'crime', u'crimin', u'crisi', u'critic', u'cruz', u'dead', u'debat', u'definit', u'direct', u'director', u'donald', u'done', u'drug', u'effort', u'email', u'employe', u'entir', u'event', u'evid', u'expos', u'fact', u'fast', u'father', u'fbi', u'first', u'florida', u'follow', u'foreign', u'former', u'foundat', u'fox', u'frisk', u'gari', u'georg', u'get', u'give', u'gop', u'gun', u'happen', u'hell', u'hide', u'hofstra', u'holt', u'huge', u'immigr', u'includ', u'interview', u'jill', u'johnson', u'kill', u'latest', u'leader', u'leav', u'lester', u'lie', u'littl', u'look', u'make', u'may', u'mayb', u'mean', u'mike', u'militari', u'million', u'minut', u'monday', u'morn', u'mouth', u'move', u'name', u'nation', u'new', u'night', u'nomine', u'north', u'offic', u'open', u'part', u'paul', u'pay', u'penc', u'person', u'point', u'polici', u'polit', u'poll', u'prais', u'prepar', u'presid', u'presidenti', u'pretti', u'privat', u'profil', u'public', u'question', u'rahami', u'receiv', u'record', u'refuge', u'rep', u'republican', u'respect', u'riot', u'run', u'russia', u'russian', u'ryan', u'said', u'sander', u'say', u'second', u'secretari', u'senat', u'set', u'shoot', u'show', u'sight', u'spend', u'stage', u'stand', u'start', u'state', u'stop', u'su', u'suggest', u'support', u'syrian', u'system', u'take', u'ted', u'televis', u'tell', u'terrifi', u'three', u'tie', u'tim', u'time', u'total', u'tri', u'trump', u'tweet', u'two', u'univers', u'video', u'vote', u'voter', u'want', u'war', u'week', u'win', u'women', u'wonder', u'work', u'wow', u'year', u'york', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'Category_left', 'Category_mainstream', 'Category_right']

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
# #3 fold best for f1_weighted is {'penalty': None, 'alpha': 10.0, 'max_iter': 2000} 
# #{'penalty': None, 'alpha': 0.1, 'max_iter': 800}

perceptron_f1 = Perceptron(penalty='None',alpha=0.1,max_iter=800,class_weight=class_weight,random_state=42)

perceptron_f1.fit(X_train, y_train)
preds = perceptron_f1.predict(X_train)
print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) # 0.86483
print "train accuracy_score: " + str(accuracy_score(y_train, preds))

preds = perceptron_f1.predict(X_test)
print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) # 0.81388
print "test accuracy_score: " + str(accuracy_score(y_test, preds))

print "confusion matrix trained with f1: \n" + str(confusion_matrix(y_test, preds))
 # [[ 21   8]
 #  [ 35 140]]

#With eta0=default
# negative weight is: 5.46982758621
# train f1_score: 0.856392547554
# train accuracy_score: 0.859427048634
# test f1_score: 0.803697130505
# test accuracy_score: 0.773936170213
# confusion matrix trained with f1: 
# [[ 55   3]
#  [ 82 236]]

#With eta0=0.0001
# train f1_score: 0.856392547554
# train accuracy_score: 0.859427048634
# test f1_score: 0.803697130505
# test accuracy_score: 0.773936170213
# confusion matrix trained with f1: 
# [[ 55   3]
#  [ 82 236]]

#Fixed Colnames
# train f1_score: 0.866274804865
# train accuracy_score: 0.851432378414
# test f1_score: 0.836078608405
# test accuracy_score: 0.816489361702
# confusion matrix trained with f1: 
# [[ 48  10]
#  [ 59 259]]

#Retuned
# train f1_score: 0.85776595971
# train accuracy_score: 0.839440373085
# test f1_score: 0.829343829016
# test accuracy_score: 0.80585106383
# confusion matrix trained with f1: 
# [[ 53   5]
#  [ 68 250]]

# PLOTS

print('\n Words contribute most to Not Mostly Factual Content')
rank_idx = np.argsort(perceptron_f1.coef_)[0]
feats_no = []
coef_no = []
for i in range(20):
    idx = rank_idx[i]
    feats_no += [colnames[idx]]
    coef_no += [perceptron_f1.coef_[0][idx]]
feats_no = feats_no[::-1]
coef_no = coef_no[::-1]
r = range(len(feats_no))
plt.barh(r, coef_no, color = "red")
plt.yticks(r, feats_no, rotation = 30)
plt.ylabel("Feature")
plt.xlabel("Coefficient Values")
plt.title("Perceptron 20 Most Important Feature that Contributes to False Content")
plt.show()


rank_idx = rank_idx[::-1]


print('\n Words contribute most to Mostly Factual Content')
feats_tru = []
coef_tru = []
for i in range(20):
    idx = rank_idx[i]
    feats_tru += [colnames[idx]]
    coef_tru += [perceptron_f1.coef_[0][idx]]
feats_tru = feats_tru[::-1]
coef_tru = coef_tru[::-1]
r = range(len(feats_tru))
plt.barh(r, coef_tru, color = "blue")
plt.yticks(r, feats_tru, rotation = 30)
plt.ylabel("Feature")
plt.xlabel("Coefficient Values")
plt.title("Perceptron 20 Most Important Feature that Contributes to True Content")
plt.show()


#print tune(X_train, y_train, 'accuracy')
# 3 fold best for accuracy is {'penalty': 'l2', 'alpha': 1000.0, 'max_iter': 1600}

# perceptron_accuracy = Perceptron(penalty='l2',alpha=1000.0,max_iter=1600,class_weight=class_weight,random_state=42)
# perceptron_accuracy.fit(X_train, y_train)
# preds = perceptron_accuracy.predict(X_train)
# print "train accuracy_score: " + str(accuracy_score(y_train, preds)) # 0.32038
# preds = perceptron_accuracy.predict(X_test)
# print "test accuracy_score: " + str(accuracy_score(y_test, preds)) # 0.3186
# print "confusion matrix trained with accuracy: \n" + str(confusion_matrix(y_test, preds))
# # [[ 28   1]
# #  [138  37]]



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




