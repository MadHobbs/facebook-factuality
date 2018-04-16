# python libraries
from string import punctuation
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# matplotlib libraries
import matplotlib.pyplot as plt

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
    tokenizer = RegexpTokenizer(r'\w+')
    input_string = input_string.lower()
    words = tokenizer.tokenize(input_string) 

    filtered_words = [w for w in words if not w in stop_words]
    filtered_words = []

    list_of_crap = ['http', 'ws', 'abcn', 'www', 'com', '2d5ogyb', '2cndgea', 'ly', '2cygp9w', '2d6hx9w', 'de', 'lz', '2ccpesl', '00', 'et', '2cyazvp', '2ck2xe1', '2d2iuqb']
 
    for w in words:
        if w not in stop_words and len(w) > 1 and w not in list_of_crap:
            filtered_words.append(w)

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
