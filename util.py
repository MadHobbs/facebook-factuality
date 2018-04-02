import pands as pd
import numpy as np

def read_dataSet():
    file_name = "facebook-fact-check.csv"
    return pd.read_csv(file_name)
def uniqueAccount():
    data = pd.read_dataSet()
    return np.unique(data.account_id)