import pandas as pd
import numpy as np

def read_dataSet():
    file_name = "facebook-fact-check.csv"
    return pd.read_csv(file_name)
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
def function():
    pass