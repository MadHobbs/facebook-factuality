"""
Author      : Shota Yasunaga, Madison Hobbs, Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 2
Description : Utilities
"""

import pandas as pd
import numpy as np
import collections
from string import punctuation
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
import Stemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import text_processing 
from sklearn.model_selection import train_test_split


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
    '''Remove rows that are videos and remove rows whose labels are "no factual 
    content" since the latter are opinion posts (not even trying to be fact.
    Code Category (news source) as three binary variables (one hot encoding)'''
    df = read_merged()
    clear_df = clear_rows(['status_message'], df)
    clear_df = clear_df.drop(clear_df[clear_df.Rating == "no factual content"].index)
    # one hot encode Category
    clear_df = pd.get_dummies(clear_df, columns=["Category"])
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
        filtered_words        -- list of lowercase "words" with 
                                 stop words and punctuation removed
    """
    stop_words = set(stopwords.words('english'))
    #input_string = input_string.encode("utf-8")
    input_string = unicode(input_string, "utf-8")
    input_string = input_string.replace("donald", "trump")
    input_string = input_string.replace("hillary", "clinton")
    tokenizer = RegexpTokenizer(pattern = "\w+")
    input_string = input_string.lower()
    words = tokenizer.tokenize(input_string)
    #stemmer = Stemmer.Stemmer('english')
    stemmer = PorterStemmer() 
    #wnl = WordNetLemmatizer()

    filtered_words = [w for w in words if not w in stop_words]
    filtered_words = []

    filtered_words = [stemmer.stem(word) for word in words]

    list_of_crap = ['utf8', 'http', 'ws', 'abcn', 'www', 'com', '2d5ogyb', '2cndgea', 'ly', '2cygp9w', '2d6hx9w', 'de', 'lz', '2ccpesl', '00', 'et', '2cyazvp', '2ck2xe1', '2d2iuqb']
 
    for w in words:
        if w not in stop_words and len(w) > 1 and w not in list_of_crap:
            filtered_words = filtered_words + [w] 

    #filtered_words = stemmer.stemWords(filtered_words)
    return filtered_words

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


######################################################################
# functions -- coding labels and producing train/test data frames    #
######################################################################

def code_truVrest():
    '''Code mostly true as 1 and mix of true/false and mostly false as 0.'''
    data = pd.read_csv('clear.csv')
    y_truVrest = data.Rating
    y_truVrest[y_truVrest != 'mostly true'] = 0 #'have false'
    y_truVrest[y_truVrest == 'mostly true'] = 1  #'mostly true'
    y_truVrest.reshape(data.Rating.shape)
    y_truVrest = np.array(y_truVrest, dtype='f')
    return y_truVrest

def make_BoW_matrix():
    '''make a matrix of the top 200 words in bag of words'''
    data = pd.read_csv('clear.csv')
    word_list = extract_dictionary(data.status_message) 
    feature_matrix = extract_feature_vectors(data.status_message, word_list)
    word_totals = feature_matrix.sum(axis=0)
    rank_idx = np.argsort(word_totals)
    rank_idx = rank_idx[::-1]
    top_200_idx = rank_idx[:200]
    feature_matrix = feature_matrix[:, tuple(top_200_idx)]
    return feature_matrix

def make_full_X():
    '''make a feature matrix out of bag of words matrix from 
    import words (lowest entropy) and metadata'''
    data = pd.read_csv('clear.csv')
    word_list = extract_dictionary(data.status_message)
    feature_matrix = extract_feature_vectors(data.status_message, word_list)
    X = feature_matrix
    y = data.Rating
    BoW = make_BoW_matrix()
    #BoW = text_processing.impWords(X,y,word_list)
    colnames = list(BoW)
    pop_data = data[['num_comments', 'num_shares', \
    'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', \
    'num_angrys', 'Category_left', 'Category_mainstream', 'Category_right']]
    pop_data_cols = ['num_reactions', 'num_comments', 'num_shares', \
    'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', \
    'num_angrys', 'Category_left', 'Category_mainstream', 'Category_right']
    X = np.hstack((BoW, pop_data))
    #X = pop_data
    colnames = colnames + pop_data_cols
    #colnames = pop_data_cols
    return X, colnames

def make_test_train():
    '''make test and train while preprocessing to normalize
    return X_train, X_test, y_train, y_test'''
    y = code_truVrest()
    X, colnames = make_full_X()
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.20)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)
    pd.DataFrame(X_train).to_csv(path_data+'X_train.csv')
    pd.DataFrame(X_test).to_csv(path_data+'X_test.csv')
    pd.DataFrame(y_train).to_csv(path_data+'y_train.csv')
    pd.DataFrame(y_test).to_csv(path_data+'y_test.csv')
    #return X_train, X_test, y_train, y_test, colnames

######################################
# Load Data -- feature extraction   ##
######################################

def load_reaction_counts(filename):
    filename = path_data + filename
    data = pd.read_csv(filename)
    reaction_list = ['num_reactions' ,'num_comments'  ,'num_shares','num_likes' ,'num_loves' ,'num_wows'  ,'num_hahas' ,'num_sads'  ,'num_angrys']    
    reactions = data[reaction_list]
    y = data.Rating.values
    X = reactions.values
    return X, y

