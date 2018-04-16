"""
ova_processing.py
Author      : Shota Yasunaga, Madison Hobbs, Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 4
Description : Script that test binary classification of
				- Non-factual VS. the rest
				- Mostly True VS. the rest
			  using Bag of Words Features
			  Objective of this exploration is to see what
			  word contribute most to Non-factual and 
			  Mostly True Posts
"""
# python libraries
import collections
from string import punctuation
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
from util import *
from bag_of_words import *

######################################################################
# MAIN #
######################################################################

    
merge_files()
write_clear()
    
data = pd.read_csv('clear.csv')
word_list = extract_dictionary(data.status_message) 
feature_matrix = extract_feature_vectors(data.status_message, word_list)
word_totals = feature_matrix.sum(axis=0)
rank_idx = np.argsort(word_totals)
rank_idx = rank_idx[::-1]
top_200_idx = rank_idx[:200]

n, d = feature_matrix.shape
new_feature_matix = np.zeros((n, 200))
index = 0

#for w in range(d):
#    if w in top_200_idx:
#        new_feature_matix[:,index] = feature_matrix[:, w]
#        index += 1

#top_200_idx = np.array(top_200_idx)
feature_matrix = feature_matrix[:, tuple(top_200_idx)]
print feature_matrix.shape

"""
rank_idx = np.argsort(word_totals)
    #rank_idx = sorted(range(len(word_totals)), key=lambda i: word_totals[i])
rank_idx = rank_idx[::-1]
print rank_idx[:20]

for key in word_list.keys():
    for idx in rank_idx[:20]:
        if word_list[key] == idx :
            print key
"""

feature_matrix = new_feature_matix
X = feature_matrix
y = data.Rating

# shuffle data (since file has tweets ordered by movie)
X, y = shuffle(X, y, random_state=0)

y_shape = y.shape

print'y_shape'
print y_shape

print 'y'
print y[:30]

# set random seed
np.random.seed(1234)

# Setup into Two different Sets of data
y_nfcVrest = np.copy(y)
y_truVrest = np.copy(y)

#y_nfcVrest = pd.DataFrame(y_nfcVrest)
#y_truVrest = pd.DataFrame(y_truVrest)

y_nfcVrest[y_nfcVrest != 'no factual content'] = 1 	#'have factual content'
y_nfcVrest[y_nfcVrest == 'no factual content'] = 0 #'no factual content'
y_nfcVrest.reshape(y_shape)
y_nfcVrest = np.array(y_nfcVrest, dtype='f')
print 'y_nfcVrest'
print y_nfcVrest[:30]

y_truVrest[y_truVrest != 'mostly true'] = 0 #'have false'
y_truVrest[y_truVrest == 'mostly true'] = 1  #'mostly true'
y_truVrest.reshape(y_shape)
y_truVrest = np.array(y_nfcVrest, dtype='f')
print 'y_truVrest'
print y_truVrest[:30]


#y_nfcVrest[y_nfcVrest != 'no factual content'] = 'have factual content'
#y_nfcVrest.reshape(y_shape)
#print 'y_nfcVrest'
#print y_nfcVrest[:30]

#y_truVrest[y_truVrest != 'mostly true'] = 'have false'
#y_truVrest.reshape(y_shape)
#print 'y_truVrest'
#print y_truVrest[:30]


# Split into training and test dataset
X_train, X_test = X[:1673], X[1673:]

y_nfcVrest_train, y_nfcVrest_test = y_nfcVrest[:1673], y_nfcVrest[1673:]
y_truVrest_train, y_truVrest_test = y_truVrest[:1673], y_truVrest[1673:]

# normalize on training set and then normalize test set 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test) 

print "\n\n Test y"
for y in y_nfcVrest_train:
	print y


# print "\n\n\n Train svm_linear_clf_nfcVrest"

# split = 3
# score = 0.0
# kf = KFold(n_splits=split)
# for train, test in kf.split(X_train):
# 	X_train_k, y_train_k = X_train[train], y_nfcVrest_train[train]
# 	X_test_k, y_test_k =  X_train[test], y_nfcVrest_train[test]
# 	svm_linear_clf_nfcVrest= SVC(kernel='linear')
# 	svm_linear_clf_nfcVrest.fit(X_train_k, y_train_k)
# 	y_nfcVrest_pred = svm_linear_clf_nfcVrest.predict(X_test_k)
# 	score += metrics.f1_score(y_test_k, y_nfcVrest_pred, average='weighted')

# print "f1 CV score"
# print score/split

# svm_linear_clf_nfcVrest = SVC(kernel='linear')
# svm_linear_clf_nfcVrest.fit(X_train, y_nfcVrest_train)
# y_nfcVrest_pred = svm_linear_clf_nfcVrest.predict(X_test)

# print svm_linear_clf_nfcVrest.coef_

# rank_idx = np.argsort(svm_linear_clf_nfcVrest.coef_)[0]
# print('rank_idx')
# print(rank_idx)

# print('\n Words contribute most to No-factual Content')
# for key in word_list.keys():
#     for idx in rank_idx[:20]:
#         if word_list[key] == idx:
#             print key

# #rank_idx = rank_idx[::-1] #reverse to Highest first

# print metrics.accuracy_score(y_nfcVrest_test, y_nfcVrest_pred)
# print metrics.f1_score(y_nfcVrest_test, y_nfcVrest_pred, average='weighted')
# print metrics.precision_score(y_nfcVrest_test, y_nfcVrest_pred, average="weighted")



print "\n\n\n Train svm_linear_clf_truVrest"
'''
split = 3
score = 0.0
kf = KFold(n_splits=split)
for train, test in kf.split(X_train):
	X_train_k, y_train_k = X_train[train], y_truVrest_train[train]
	X_test_k, y_test_k =  X_train[test], y_truVrest_train[test]
	svm_linear_clf_truVrest= SVC(kernel='linear')
	svm_linear_clf_truVrest.fit(X_train_k, y_train_k)
	y_truVrest_pred = svm_linear_clf_truVrest.predict(X_test_k)
	score += metrics.f1_score(y_test_k, y_truVrest_pred, average='weighted')

print "f1 CV score"
print score/split
'''

svm_linear_clf_truVrest= SVC(kernel='linear')
svm_linear_clf_truVrest.fit(X_train, y_truVrest_train)
y_truVrest_pred = svm_linear_clf_truVrest.predict(X_test)

print svm_linear_clf_truVrest.coef_

rank_idx = np.argsort(svm_linear_clf_truVrest.coef_)[0]
rank_idx = rank_idx[::-1]
print rank_idx

print('\n Words contribute most to Mostly Factual Content')
for key in word_list.keys():
    for idx in rank_idx[:50]:
        if word_list[key] == idx:
            print key

rank_idx = rank_idx[::-1]
print rank_idx

print('\n Words contribute most to No Factual Content')
for key in word_list.keys():
    for idx in rank_idx[:50]:
        if word_list[key] == idx:
            print key

print metrics.accuracy_score(y_truVrest_test, y_truVrest_pred)
print metrics.f1_score(y_truVrest_test, y_truVrest_pred, average='weighted')
print metrics.precision_score(y_truVrest_test, y_truVrest_pred, average="weighted")


print "---- using popularity counts (num likes, shares, comments etc ...) ----- "
# get popularity counts
pop_data = data[['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]
X = np.hstack((feature_matrix, pop_data))
y = data.Rating
print X.shape 
print y.shape
# shuffle data (since file has tweets ordered by movie)
X, y = shuffle(X, y, random_state=0)

# set random seed
np.random.seed(1234)

# Setup into Two different Sets of data
y_nfcVrest = np.copy(y)
y_truVrest = np.copy(y)

#y_nfcVrest = pd.DataFrame(y_nfcVrest)
#y_truVrest = pd.DataFrame(y_truVrest)

y_nfcVrest[y_nfcVrest != 'no factual content'] = 1 	#'have factual content'
y_nfcVrest[y_nfcVrest == 'no factual content'] = 0 #'no factual content'
y_nfcVrest.reshape(y_shape)
y_nfcVrest = np.array(y_nfcVrest, dtype='f')
print 'y_nfcVrest'
print y_nfcVrest[:30]

y_truVrest[y_truVrest != 'mostly true'] = 0 #'have false'
y_truVrest[y_truVrest == 'mostly true'] = 1  #'mostly true'
y_truVrest.reshape(y_shape)
y_truVrest = np.array(y_nfcVrest, dtype='f')
print 'y_truVrest'
print y_truVrest[:30]

# Split into training and test dataset
X_train, X_test = X[:1673], X[1673:]

y_nfcVrest_train, y_nfcVrest_test = y_nfcVrest[:1673], y_nfcVrest[1673:]
y_truVrest_train, y_truVrest_test = y_truVrest[:1673], y_truVrest[1673:]

# normalize on training set and then normalize test set 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test) 

svm_linear_clf_truVrest= SVC(kernel='linear')
svm_linear_clf_truVrest.fit(X_train, y_truVrest_train)
y_truVrest_pred = svm_linear_clf_truVrest.predict(X_test)

print svm_linear_clf_truVrest.coef_

rank_idx = np.argsort(svm_linear_clf_truVrest.coef_)[0]
rank_idx = rank_idx[::-1]
print rank_idx

print('\n Words contribute most to Mostly Factual Content')
for key in word_list.keys():
    for idx in rank_idx[:50]:
        if word_list[key] == idx:
            print key

rank_idx = rank_idx[::-1]
print rank_idx

print('\n Words contribute most to No Factual Content')
for key in word_list.keys():
    for idx in rank_idx[:50]:
        if word_list[key] == idx:
            print key

print metrics.accuracy_score(y_truVrest_test, y_truVrest_pred)
print metrics.f1_score(y_truVrest_test, y_truVrest_pred, average='weighted')
print metrics.precision_score(y_truVrest_test, y_truVrest_pred, average="weighted")









