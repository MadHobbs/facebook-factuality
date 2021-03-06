"""
Author      : Shota Yasunaga, Madison Hobbs, Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 2
Description : Project Data Exploration
"""
# python libraries
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.svm import SVC

# libraries for the information gain
import heapq as hp

# libraries specific to project
from util import *
from bag_of_words import *
from sklearn import metrics

import sys # modify path because these files are in a different directory
sys.path.insert(0, '../models')
from validations import *
from dtree import *

###############################
## Entropy Estimation Stuff ###
###############################
def impWords(X,y,word_list,search_space = 300, max_bag = 200, num_appear_limit = 2):
    '''
    input: 
    X ... feature vector (n,m)
    y ... class          (n,)
    word_list ... dictionary of words with index
    search_space ... How many of the words in word_list you want to consider
    max_bag      ... The maximum of the returned words list
    num_apprea_limit ... Words have to appreat more than num_apprear_limit times in the posts

    output:
    panda datafreme with the shape (num_sampels, words) with the label of the words
    '''
    #print '-------------------------'
    n_features = len(X[0])
    n_classes  = len(np.unique(y))
    tree = Tree(n_features,n_classes)
    information_gain_list = np.zeros(len(X[0]))

    for i in range(len(X)):
        information_gain_list[i] = tree._information_gain(X[:,i],y)[0]

    feature_index = hp.nlargest(500, range(len(information_gain_list)), information_gain_list.take)
    print (feature_index)
    print ('---------------y------------------')

    #print feature_index
    #print '---------------y------------------'
        
    print ('=======================================')

    # For wordMap
    info_gain_dic = {} 
    feature_dic = {}
    n = 0 
    for i in feature_index:
        for key, value in word_list.iteritems():    # for name, age in list.items():  (for Python 3.x)
            if value == i:
                if sum(X[:,i]) > num_appear_limit:
                    n+=1
                    feature_dic[key] = X[:,i]
                    #rint (key)
                    #print (sum(X[:,i]))
                    info_gain_dic[key] = information_gain_list[value]
                if n == max_bag:
                    #print ('there are ', n, 'bag of words')
                    return pd.DataFrame(data=feature_dic)

    # For wordMap
    #print 'there are ', n, 'bag of words'
    return pd.DataFrame(data=feature_dic)
# Due to my laziness, I made another function that returns something slightly different
def impWords_info_dic(X,y,word_list,search_space = 300, max_bag = 200, num_appear_limit = 2):
    '''
    input: 
    X ... feature vector (n,m)
    y ... class          (n,)
    word_list ... dictionary of words with index
    search_space ... How many of the words in word_list you want to consider
    max_bag      ... The maximum of the returned words list
    num_apprea_limit ... Words have to appreat more than num_apprear_limit times in the posts

    output:
    panda datafreme with the shape (num_sampels, words) with the label of the words
    '''
    #print '-------------------------'
    n_features = len(X[0])
    n_classes  = len(np.unique(y))
    tree = Tree(n_features,n_classes)
    information_gain_list = np.zeros(len(X[0]))

    for i in range(len(X)):
        information_gain_list[i] = tree._information_gain(X[:,i],y)[0]

    feature_index = hp.nlargest(500, range(len(information_gain_list)), information_gain_list.take)
    # print feature_index
    # print '---------------y------------------'

    #print feature_index
    #print '---------------y------------------'
        
    # print '======================================='

    # For wordMap
    info_gain_dic = {} 
    feature_dic = {}
    n = 0 
    for i in feature_index:
        for key, value in word_list.iteritems():    # for name, age in list.items():  (for Python 3.x)
            if value == i:
                if sum(X[:,i]) > num_appear_limit:
                    n+=1
                    info_gain_dic[key] = information_gain_list[value]
                if n == max_bag:
                    # print 'there are ', n, 'bag of words'
                    return info_gain_dic
    # For wordMap

    #print 'there are ', n, 'bag of words'
    return info_gain_dic

def wordMap(info_gain_dic):
    print (info_gain_dic)
    for word, gain in info_gain_dic.iteritems():
        print ((word + " ") * int(gain*1000))
    
######################################################################
# MAIN #
######################################################################

def main():
    
    merge_files()
    write_clear()
    
    data = pd.read_csv('clear.csv')
    word_list = extract_dictionary(data.status_message) 
    feature_matrix = extract_feature_vectors(data.status_message, word_list)
    word_totals = feature_matrix.sum(axis=0)
    rank_idx = np.argsort(word_totals)
    rank_idx = rank_idx[::-1]
    print (rank_idx[:200])

    # for key in word_list.keys():
    #     for idx in rank_idx[:200]:
    #         if word_list[key] == idx :
    #             print key

    X = feature_matrix
    # print X.shape
    y = data.Rating
    # print np.unique(y)

    X_train, X_test, y_train, y_test, colnames = make_test_train()
    # define score
    f1_scorer = make_scorer(f1_score, average='samples')

    #hyperparameter selection
    '''
    parameters = {'kernel':('linear', 'rbf'), 'C':[10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X, y)
    print pd.DataFrame.from_dict(clf.cv_results_)
    print clf.best_params_
    print clf.best_score_
    '''

    # clf = SVC(kernel='linear', C = 0.1)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # print clf.coef_

    # print metrics.f1_score(y_test, y_pred, average='weighted')

    # print "training error"
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_train)
    # print metrics.f1_score(y_train, y_pred, average='weighted')

    # pandaman=impWords(X,y,word_list)
    # print pandaman
    info_gain_dic = impWords_info_dic(X,y,word_list)
    wordMap(info_gain_dic)
    
if __name__ == "__main__" :
   main()



