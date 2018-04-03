import pandas as pd
import numpy as np

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

def write_clear()
    df = read_merged()
    clear_df = clear_rows(['status_message'], df)
    clear_df.to_csv('clear.csv')

