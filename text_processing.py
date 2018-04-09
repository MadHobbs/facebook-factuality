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

# libraries specific to project
from util import *
from bag_of_words import *
from validations import *
from sklearn import metrics

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
    print X.shape
    y = data.Rating
    print np.unique(y)

    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=42)
    # set random seed
    np.random.seed(42)

    X_train, X_test = X[:1950], X[1950:]
    y_train, y_test = y[:1950], y[1950:]

    # define score
    f1_scorer = make_scorer(f1_score, average='samples')

    # hyperparameter selection
    parameters = {'kernel':('linear', 'rbf'), 'C':[10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X, y)
    #print pd.DataFrame.from_dict(clf.cv_results_)
    print clf.best_params_

    #clf = SVC(kernel='linear')
    #clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)

    #print clf.coef_

    #print metrics.accuracy_score(y_test, y_pred)
    #print metrics.f1_score(y_test, y_pred, average='weighted')
    #print metrics.precision_score(y_test, y_pred, average="weighted")


if __name__ == "__main__" :
    main()



