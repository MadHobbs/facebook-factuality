"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2018 Feb 5
Description : Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

######################################################################
# functions
######################################################################

def cv_performance(clf, train_data, kfs):
    """
    Determine classifier performance across multiple trials using cross-validation
    
    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold
    
    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """
    n_trials = len(kfs)
    n_folds = kfs[0].n_splits
    scores = np.zeros((n_trials, n_folds))
    
    ### ========== TODO : START ========== ###
    # part b: run multiple trials of CV
    for k in range(n_trials):
        kf = kfs[k]
        scores[k] =cv_performance_one_trial(clf,train_data,kf)
    
    ### ========== TODO : END ========== ###
    
    return scores


def cv_performance_one_trial(clf, train_data, kf) :
    """
    Compute classifier performance across multiple folds using cross-validation
    
    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold
    
    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """
    
    scores = np.zeros(kf.n_splits)
    
    ### ========== TODO : START ========== ###
    # part b: run one trial of CV
    n_folds = kf.n_splits
    i = 0
    for train_index,test_index in kf.split(train_data.X):
        clf.fit(train_data.X[train_index],train_data.y[train_index])
        prediction = clf.predict(train_data.X[test_index])
        scores[i] = metrics.accuracy_score(prediction, train_data.y[test_index], normalize = True)
        i += 1

    ### ========== TODO : END ========== ###
    
    return scores


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)
    
    #========================================
    # load data
    train_data = load_data("phoneme_train.csv")
    
    ### ========== TODO : START ========== ###
    # part a: is data linearly separable?
    X = train_data.X
    y = train_data.y

    clf = Perceptron(max_iter = 10000)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print 'train_error %f' %train_error

    ### ========== TODO : END ========== ###
        
    ### ========== TODO : START ========== ###
    # part c-d: compare classifiers
    # make sure to use same folds across all runs
    def calc_std(scores):
        """Calculate standard deviation for each kFold arross all the values
        """
        return np.std(scores.flatten())



    def do_kfs(train_data):
        """ do_kfs calculate k-Fold cross validation score. 
            return kf_scores
        """
        kfs = []
        for i in range(4):
            kfs.append(model_selection.KFold(n_splits=10, shuffle = True, random_state= i))

        clfs = [DummyClassifier(strategy="most_frequent"), Perceptron(max_iter = 10000),LogisticRegression(penalty = 'l1',C=10000)]
        clfs_names = ["DummyClassifier", "Perceptron", "LogisticRegression"]
        kf_scores = []
        for clf in clfs:
            print clf
            kf_scores.append( cv_performance(clf, train_data, kfs))

        for j in range(len(clfs)):
            print clfs_names[j] ,":",np.mean(kf_scores[j])

        return kf_scores
    
    ########################
    ### w/o preprocessing ##
    ########################
    regular_scores = do_kfs(train_data)


    ########################
    ### mean subtraction ###
    ########################
    standard_data = Data()
    #train_stdX = X_standard.mean(axis=0)

    standard_data.X = preprocessing.scale(train_data.X)
    standard_data.y = train_data.y
    standard_scores = do_kfs(standard_data)


    # Run the t-test
    name = ['Dummy','Perceptron','LogisticRegression','STD Dummy','STD Perceptron','STD LogisticRegression']
    score_list = regular_scores+standard_scores
    for clf1 in range(len(score_list)):
        for clf2 in range(len(score_list)):
            if clf2 > clf1:
                print name[clf1],'vs',name[clf2], '=', stats.ttest_rel(score_list[clf1].flatten(),score_list[clf2].flatten())




    
    
    ### ========== TODO : END ========== ###
    
    ### ========== TODO : START ========== ###
    # part e: plot
    ########################
    ### Plotting errors  ###
    ########################
    # Code obtained from https://matplotlib.org/examples/api/barchart_demo.html and being modified

    percep_means = (np.mean(regular_scores[1]),np.mean(standard_scores[1]))
    percep_stds = (calc_std(regular_scores[1]), calc_std(standard_scores[1]))
    log_reg_means = (np.mean(regular_scores[2]),np.mean(standard_scores[2]))
    log_reg_stds  = (calc_std(regular_scores[2]), calc_std(standard_scores[2]))
    print 'percep_stds',percep_stds
    print 'log_reg_stds', log_reg_stds
    ind = np.arange(2)
    width = 0.20
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, percep_means, width, color='r', yerr=percep_stds)

    rects2 = ax.bar(ind +width, log_reg_means, width, color='b', yerr=log_reg_stds)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Preprocessing and Classifier')
    ax.set_xticks(ind+width/2)
    ax.set_xticklabels(('No preprocessing', 'Standardization'))

    ax.legend((rects1[0], rects2[0]), ('Perceptron', 'Logistic Regression'))

    ## Add Dummy line
    dummy_mean = np.mean(regular_scores[0])
    dummy_std =  calc_std(regular_scores[0])
    dummy_top = dummy_mean+dummy_std
    dummy_bottom = dummy_mean- dummy_std
    print 'dummy_std',dummy_std
    plt.plot([0,1.25],[dummy_top,dummy_top],color ='y',linestyle='--',linewidth =2)
    plt.plot([0,1.25], [dummy_mean,dummy_mean],color='y',linestyle='-',linewidth=2)
    plt.plot([0,1.25], [dummy_bottom,dummy_bottom],color = 'y', linestyle='--',linewidth=2)
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%.3f' % float(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.show()
    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()