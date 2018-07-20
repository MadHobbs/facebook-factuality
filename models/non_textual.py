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
from soybeans import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


def main():
    ## Split into train/test
    # TODO: in the fufture, we should fold them later
    X,y = util.load_reaction_counts('merged.csv')
    enum_dic = {'no factual content':0, 'mostly true':1, 'mostly false':2, 'mixture of true and false':3}
    for i in range(len(y)):
        y[i] = enum_dic[y[i]]
    sss = StratifiedShuffleSplit(n_splits=1,test_size =0.3,random_state=0)
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]



    ################################################################
    ## Predict factuality from number of                          ##
    ## shares, comments, likes, loves, wows,hahas, sads, angrys   ##
    ################################################################
    # -- This is about if people respond to posts depending on    ## 
    # -- the factuality of the post. (this might be some          ##
    # -- underlying cusation of media intention)                  ##
    ################################################################
    num_classes = 4
    loss_func_list = [hamming_losses,sigmoid_losses,logistic_losses]
    R_list = [generate_output_codes(num_classes, 'ova'),generate_output_codes(num_classes, 'ovo')]
    code_itr = iter(['ova','ovo']*3)

    print 'classifying...'
    for loss_func in loss_func_list :
        for R in R_list : 
    #   train a multiclass SVM on training data and evaluate on test data
    #   setup the binary classifiers using the specified parameters from the handout
            # clf = MulticlassSVM(R = R, kernel='poly', degree = 4, coef0 = 1, gamma = 1.0)
            clf = MulticlassSVM(R = R, kernel='linear')
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test, loss_func=loss_func)
            print str(loss_func)
            print code_itr.next()
            num_errors = sum(pred != y_test)
            print "number of erros: " + str(num_errors) 
            print 'Accuracy: ', (1.0 - (num_errors/float(len(y))))
            print '\n\n'

######################################################################
## __main__                                                         ##
######################################################################


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

    # run on smaller subset
    X_train, X_test = X[:836], X[836:1045]
    y_train, y_test = y[:836], y[836:1045]

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

    # minority class gets larger weight (as per example in class)
    weight = (1-fracNeg)/float(fracNeg)
    class_weight = {1:1, -1:weight}
    print "weight for negatives : " + str((1-fracNeg)/float(fracNeg))

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