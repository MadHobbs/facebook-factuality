"""
Author      : Shota Yasunaga, Madison Hobbs, Justin Lauw
Class       : HMC CS 158
Date        : 2018 April 2
Description : Project Data Exploration
"""

import pandas as pd
import numpy as np
from string import punctuation

def read_dataSet():
    file_name = "facebook-fact-check.csv"
    return pd.read_csv(file_name)

def read_merged():
    return pd.read_csv("merged.csv")

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
    fb_fact_check = pd.read_csv('facebook-fact-check.csv')
    fb_statuses = pd.read_csv('facebook_statuses.csv')
    fb_statuses['account_id'], fb_statuses['post_id'] = fb_statuses['status_id'].str.split('_', 1).str

    fb_fact_check[['account_id', 'post_id']] = fb_fact_check[['account_id', 'post_id']].astype(int)
    fb_statuses[['account_id', 'post_id']] = fb_statuses[['account_id', 'post_id']].astype(int)

    fb_fact_check = fb_fact_check.merge(fb_statuses, how='inner', left_on=['account_id','post_id'], right_on = ['account_id', 'post_id'])
    fb_fact_check.to_csv("merged.csv")

def write_clear():
    df = read_merged()
    clear_df = clear_rows(['status_message'], df)
    clear_df.to_csv('clear.csv')

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
    
