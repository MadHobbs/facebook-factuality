"""
Author      : Shota Yasunaga, Madison Hobbs, Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 2
Description : Project Data Exploration
"""

import pandas as pd
import numpy as np
from string import punctuation
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# path_data = "../data/"
path_data = ""

def read_dataSet():
    file_name = path_data + "facebook-fact-check.csv"
    return pd.read_csv(file_name)

def read_merged():
    return pd.read_csv(path_data+"merged.csv")

def uniqueAccount():
    data = read_dataSet()
    return np.unique(data.account_id)

# postInfo
# return a lists of tuples containing (page_id, post_id)
# removes https://www.facebook.com/FreedomDailyNews

def postInfo():
    data = read_dataSet()
    resultList = []
    for i in range(len(data.account_id)):
        if data.account_id[i]!= 440106476051475:
            tup = (data.account_id[i], data.post_id[i])
            resultList.append(tup)
    return resultList


# column: List of column that you want to be non empty
# df: pandas dataFrame
# depreciate logical:  if you want it to be both not empty? one of the?
#           ex) and if you want both to be true
# return the rows that correspond 
def clear_rows(column_list,  df):
    if len(column_list) < 2:
        return df[pd.notnull(df[column_list[0]])]
    else:
        for column in column_list:
            df = df[pandas.notnull(df[column])]
        return df

######################################################################
# load data
######################################################################

def merge_files():
    fb_fact_check = pd.read_csv(path_data+'facebook-fact-check.csv')
    fb_statuses = pd.read_csv(path_data+ 'facebook_statuses.csv')
    fb_statuses['account_id'], fb_statuses['post_id'] = fb_statuses['status_id'].str.split('_', 1).str

    fb_fact_check[['account_id', 'post_id']] = fb_fact_check[['account_id', 'post_id']].astype(int)
    fb_statuses[['account_id', 'post_id']] = fb_statuses[['account_id', 'post_id']].astype(int)

    fb_fact_check = fb_fact_check.merge(fb_statuses, how='inner', left_on=['account_id','post_id'], right_on = ['account_id', 'post_id'])
    fb_fact_check.to_csv("merged.csv")

def write_clear():
    df = read_merged()
    clear_df = clear_rows(['status_message'], df)
    clear_df.to_csv(path_data+'clear.csv')


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string) :
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()

def extract_dictionary(df_column) :
    """
    Given a dataframe, builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        df_column    -- column of pandas dataframe
                        (list of strings)
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}

    wordsL = []
    for post in df_column:
        wordsL.extend(extract_words(post))

    index = 0
    for word in wordsL :
        if not word in word_list :
            word_list[word] = index
            index += 1

    return word_list

def extract_feature_vectors(df_column, word_list) :
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        df_column         -- list of strings, column of dataframe
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = len(df_column)
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    row = 0
    for post in df_column:
        postWordsL = extract_words(post)
        for word in postWordsL:
            if word in word_list :
                column = word_list[word]
                feature_matrix[row][column] = 1
        row += 1
    
    return feature_matrix


######################################
# Load Data -- feature extraction   ##
######################################

def load_reation_counts(filename):
    filename = path_data + filename
    data = pd.read_csv(filename)
    reaction_list = ['num_reactions' ,'num_comments'  ,'num_shares','num_likes' ,'num_loves' ,'num_wows'  ,'num_hahas' ,'num_sads'  ,'num_angrys']    
    reactions = data[reaction_list]
    y = data.Rating
    X = reactions.values
    return X, y

####################
# Cross Validation #
####################

def cv_performance(clf, X, y, kf, metric="accuracy") :
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
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
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
