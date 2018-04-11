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
from bag_of_words import *
from validations import *
from sklearn import metrics
import util


def main():
    X,y = util.load_reaction_counts('merged.csv')
    print X
    print '------------------------'
    print y
    ################################################################
    ## Predict factuality from number of                          ##
    ## shares, comments, likes, loves, wows,hahas, sads, angrys   ##
    ################################################################
    # -- This is about if people respond to posts depending on    ## 
    # -- the factuality of the post. (this might be some          ##
    # -- underlying cusation of media intention)                  ##
    ################################################################


    ################################################################
    ## Predict factuality from bag of words AND number of         ##
    ## shares, comments, likes, loves, wows,hahas, sads, angrys   ##
    ################################################################
    print '------------------------'
    print 'bag of words and popularity as predictors'
    print '------------------------'
    data = pd.read_csv('clear.csv')
    word_list = extract_dictionary(data.status_message) 
    feature_matrix = extract_feature_vectors(data.status_message, word_list)
    word_totals = feature_matrix.sum(axis=0)
    rank_idx = np.argsort(word_totals)

    X = data[['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]
    y = data['Rating']
    X = np.hstack((X, feature_matrix))
    
    # shuffle data (since file has tweets ordered by movie)
    X, y = shuffle(X, y, random_state=1234)
    # set random seed
    np.random.seed(42)

    X_train, X_test = X[:1673], X[1673:]
    y_train, y_test = y[:1673], y[1673:]

    # define score
    f1_scorer = make_scorer(f1_score, average='weighted')

    #hyperparameter selection
    '''
    parameters = {'kernel':('linear', 'rbf'), 'C':[10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    print pd.DataFrame.from_dict(clf.cv_results_)
    print clf.best_params_
    print clf.best_score_
    '''

    print "SVC with linear kernel, C = 0.1"

    clf = SVC(kernel='linear', C = 0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print "test error"
    print metrics.f1_score(y_test, y_pred, average='weighted')

    print "training error"
    y_pred = clf.predict(X_train)
    print metrics.f1_score(y_train, y_pred, average='weighted')

    print "SVC with poly kernel, degree 4"

    clf = SVC(kernel='poly', degree = 4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "test error"
    print metrics.f1_score(y_test, y_pred, average='weighted')

    print "training error"
    y_pred = clf.predict(X_train)
    print metrics.f1_score(y_train, y_pred, average='weighted')


if __name__ == "__main__" :
    main()