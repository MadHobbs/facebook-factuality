"""
Author      : Shota Yasunaga, Madison Hobbs, Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 2
Description : Project Data Exploration
"""
# python libraries
import collections
from string import punctuation
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import metrics

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
from util import *

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
    #rank_idx = sorted(range(len(word_totals)), key=lambda i: word_totals[i])
    rank_idx = rank_idx[::-1]
    print rank_idx[:20]

    for key in word_list.keys():
        for idx in rank_idx[:20]:
            if word_list[key] == idx :
                print key

    X = feature_matrix
    y = data.Rating

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=0)
    # set random seed
    np.random.seed(1234)

    X_train, X_test = X[:1950], X[1950:]
    y_train, y_test = y[:1950], y[1950:]

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print clf.coef_
    print np.unique(y)

    print metrics.accuracy_score(y_test, y_pred)


if __name__ == "__main__" :
    main()



