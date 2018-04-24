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
X_train = pd.read_csv("X_train.csv")
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

colnames = [u'2016', u'actual', u'address', u'aid', u'alleg', u'also', u'america', u'american', u'among', u'back', u'bad', u'barack', u'battleground', u'berni', u'best', u'better', u'bill', u'black', u'board', u'break', u'bush', u'busi', u'bust', u'call', u'came', u'campaign', u'candid', u'carolina', u'caught', u'chariti', u'check', u'children', u'chris', u'christi', u'citi', u'claim', u'clinton', u'cnn', u'come', u'communiti', u'compani', u'controversi', u'countri', u'court', u'crime', u'crimin', u'crisi', u'critic', u'cruz', u'dead', u'debat', u'definit', u'direct', u'director', u'donald', u'done', u'drug', u'effort', u'email', u'employe', u'entir', u'event', u'evid', u'expos', u'fact', u'fast', u'father', u'fbi', u'first', u'florida', u'follow', u'foreign', u'former', u'foundat', u'fox', u'frisk', u'gari', u'georg', u'get', u'give', u'gop', u'gun', u'happen', u'hell', u'hide', u'hofstra', u'holt', u'huge', u'immigr', u'includ', u'interview', u'jill', u'johnson', u'kill', u'latest', u'leader', u'leav', u'lester', u'lie', u'littl', u'look', u'make', u'may', u'mayb', u'mean', u'mike', u'militari', u'million', u'minut', u'monday', u'morn', u'mouth', u'move', u'name', u'nation', u'new', u'night', u'nomine', u'north', u'offic', u'open', u'part', u'paul', u'pay', u'penc', u'person', u'point', u'polici', u'polit', u'poll', u'prais', u'prepar', u'presid', u'presidenti', u'pretti', u'privat', u'profil', u'public', u'question', u'rahami', u'receiv', u'record', u'refuge', u'rep', u'republican', u'respect', u'riot', u'run', u'russia', u'russian', u'ryan', u'said', u'sander', u'say', u'second', u'secretari', u'senat', u'set', u'shoot', u'show', u'sight', u'spend', u'stage', u'stand', u'start', u'state', u'stop', u'su', u'suggest', u'support', u'syrian', u'system', u'take', u'ted', u'televis', u'tell', u'terrifi', u'three', u'tie', u'tim', u'time', u'total', u'tri', u'trump', u'tweet', u'two', u'univers', u'video', u'vote', u'voter', u'want', u'war', u'week', u'win', u'women', u'wonder', u'work', u'wow', u'year', u'york', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'Category_left', 'Category_mainstream', 'Category_right']

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



# #print tune(X_train, y_train, 'f1_weighted')
# 3 fold best for f1_weighted is {'penalty': None, 'alpha': 0.01, 'max_iter': 1200}

perceptron_f1 = Perceptron(penalty='None',alpha=0.001,max_iter=1000,class_weight=class_weight,random_state=42)
perceptron_f1.fit(X_train, y_train)
preds = perceptron_f1.predict(X_train)
print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) # 0.86483
preds = perceptron_f1.predict(X_test)
print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) # 0.81388
print "confusion matrix trained with f1: \n" + str(confusion_matrix(y_test, preds))
 # [[ 21   8]
 #  [ 35 140]]

feats = ["Category_mainstream", "num_shares", "Category_right", "num_wows", "num_likes", "num_reactions", \
"num_comments", "num_angrys", "num_hahas", "num_sads", "num_loves", "donald", \
"trump", "Category_left", "clinton", "president", "debate", "says", "video", \
"republican", "said", "america", "george", "americans", "racist"]

imps = [0.135356, 0.099026, 0.098160, 0.066903, 0.064147, 0.061870, 0.061593, 0.050732, \
0.047292, 0.045151, 0.044853, 0.019866, \
0.015771, 0.013685, 0.008299, 0.005285, 0.004969, 0.004541, 0.004141, 0.004071, \
0.003881, 0.003496, 0.003278, 0.003093, 0.003005]


rank_idx = np.argsort(svm_linear_clf_truVrest.coef_)[0]
feats
print('\n Words contribute most to Not Mostly Factual Content')
for key in word_list.keys():
    for idx in rank_idx[:50]:
        if word_list[key] == idx:
            print key

r = range(len(feats))
plt.bar(r, imps, color = "blue")
plt.xticks(r, feats, rotation = 70)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forests Feature Importance")



rank_idx = rank_idx[::-1]
print rank_idx

print('\n Words contribute most to Mostly Factual Content')
for key in word_list.keys():
    for idx in rank_idx[:50]:
        if word_list[key] == idx:
            print key

rank_idx = rank_idx[::-1]
print rank_idx

r = range(len(feats))
plt.bar(r, imps, color = "blue")
plt.xticks(r, feats, rotation = 70)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forests Feature Importance")


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




