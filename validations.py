# python libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ours
from util import *

####################
# Cross Validation #
####################

def lineplot(x, y, label):
    """
    Make a line plot.
    
    Parameters
    --------------------
        x            -- list of doubles, x values
        y            -- list of doubles, y values
        label        -- string, label for legend
    """
    
    xx = range(len(x))
    plt.plot(xx, y, linestyle='-', linewidth=2, label=label)
    plt.xticks(xx, x)   

def check_overfit(clf, metric, *args):
    '''Given a classifier function and'''
    y = code_truVrest()
    X = make_full_X()
    X, y = shuffle(X, y, random_state=42)
    n = len(y)
    step = n/10
    train_scores = []
    test_scores = []
    for i in range(step, n-step, step):
        print i
        X_train, X_test = X[:i], X[i:]
        y_train, y_test = y[:i], y[i:]
        # normalize on training set and then normalize test set 
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train) 
        X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        train_preds = clf.predict(X_train)
        train_scores.append(metric(y_train, train_preds, *args))
        test_preds = clf.predict(X_test)
        test_scores.append(metric(y_test, test_preds, *args)) 
    
    x_axis = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    plt.plot(x_axis, train_scores, 'b', label = "training score")
    plt.plot(x_axis, test_scores, 'g', label = "test score")
    plt.legend()
    plt.xlabel("fraction of data used to train model")
    plt.ylabel(str(metric))
    plt.show()
    

