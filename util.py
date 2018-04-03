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

######################################################################
# load data
######################################################################
fb_fact_check = pd.read_csv('facebook-fact-check.csv')
fb_statuses = pd.read_csv('facebook_statuses.csv')
fb_statuses['account_id'], fb_statuses['post_id'] = fb_statuses['status_id'].str.split('_', 1).str

fb_fact_check[['account_id', 'post_id']] = fb_fact_check[['account_id', 'post_id']].astype(int)
fb_statuses[['account_id', 'post_id']] = fb_statuses[['account_id', 'post_id']].astype(int)

fb_fact_check = fb_fact_check.merge(fb_statuses, how='inner', left_on=['account_id','post_id'], right_on = ['account_id', 'post_id'])
fb_fact_check.to_csv("merged.csv")