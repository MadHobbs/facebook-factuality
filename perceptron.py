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

colnames = [u'1', u'2016', u'a', u'abcn', u'about', u'after', u'aid', u'american', u'an', u'and', u'back', u'bad', u'barack', u'been', u'better', u'bill', u'black', u'breaking', u'bush', u'busi', u'but', u'call', u'campaign', u'candid', u'candidate', u'candidates', u'carolina', u'check', u'children', u'chri', u'chris', u'citi', u'city', u'claim', u'claims', u'clinton', u'come', u'comments', u'commun', u'community', u'countri', u'country', u'critic', u'cruz', u'debat', u'debate', u'debates', u'democratic', u'did', u'do', u'donald', u'drug', u'drugs', u'dure', u'email', u'employe', u'event', u'fact', u'first', u'florida', u'follow', u'for', u'foundat', u'frisk', u'from', u'georg', u'george', u'get', u'gets', u'getting', u'give', u'gop', u'great', u'gun', u'ha', u'happen', u'happening', u'he', u'hi', u'hofstra', u'http', u'huge', u'i', u'in', u'includ', u'interview', u'j', u'johnson', u'kill', u'latest', u'leader', u'leaders', u'leav', u'leave', u'looking', u'm', u'make', u'manager', u'may', u'mayb', u'maybe', u'mike', u'militari', u'military', u'million', u'morn', u'morning', u'most', u'my', u'nation', u'national', u'new', u'nomine', u'nominee', u'north', u'now', u'of', u'off', u'offic', u'on', u'open', u'over', u'own', u'paul', u'penc', u'pence', u'person', u'point', u'polici', u'policy', u'polit', u'political', u'poll', u'polls', u'possibility', u'presid', u'president', u'presidenti', u'presidential', u'pretti', u'pretty', u'question', u'rahami', u'record', u'refuge', u'released', u'rep', u'reporter', u'republican', u'respect', u'russia', u's', u'said', u'say', u'says', u'senat', u'senator', u'set', u'shoot', u'shooting', u'show', u'stage', u'stand', u'start', u'state', u'states', u'stop', u'suggest', u'support', u't', u'take', u'ted', u'televis', u'that', u'the', u'thi', u'to', u'total', u'tri', u'trump', u'try', u'tweet', u'two', u'univers', u'university', u'up', u'used', u'video', u'voter', u'voters', u'wa', u'war', u'week', u'while', u'who', u'win', u'with', u'ws', u'year', u'you', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'Category_left', 'Category_mainstream', 'Category_right']

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




