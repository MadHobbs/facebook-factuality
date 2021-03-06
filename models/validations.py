# python libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import metrics, preprocessing
from sklearn.dummy import DummyClassifier
# ours
import sys
sys.path.insert(0, '../data+wrangling')
import util

####################
# Cross Validation #
####################
def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    scores = []
    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score) :
            scores.append(score)
    return np.array(scores).mean()

def select_param_linear(X, y, kf, metric="accuracy", plot=True, class_weight = {1:1, -1:1}) :
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximizes' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure
        plot   -- boolean, make a plot
        class_weight -- class weights if we want to do a weighted SVC. 
                         Defaults to not weighting any class in particular.
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
        ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    scores = [0 for _ in xrange(len(C_range))] # dummy values, feel free to change
    # search over all C values
    for c in range (len(C_range)):
        clf = SVC(kernel = 'linear', C = C_range[c], class_weight=class_weight)
        scores[c] = cv_performance(clf,X,y,kf,metric = metric)

    # which C gives the best score?
    max_ind = np.argmax(scores)
    if plot:
        lineplot(C_range, scores, metric)
    
    return C_range[max_ind]

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
    y = util.code_truVrest()
    X, colnames = util.make_full_X()
    X, y = shuffle(X, y, random_state=42)
    n = len(y)
    step = n/10
    train_scores = []
    test_scores = []
    dummy_scores =[]
    dummy = DummyClassifier(strategy = "most_frequent")
    for i in range(step, n-step, step):
        print (i)
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

        dummy.fit(X_train, y_train)
        test_preds = dummy.predict(X_test)
        dummy_scores.append(metric(y_test, test_preds, *args))
    
    x_axis = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    plt.plot(x_axis, train_scores, 'b', label = "training score")
    plt.plot(x_axis, test_scores, 'g', label = "test score")
    plt.plot(x_axis, dummy_scores, 'k--', label = "baseline (majority vote) score")
    plt.legend()
    plt.xlabel("fraction of data used to train model")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.show()
    

