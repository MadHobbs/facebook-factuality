from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
# modify path because these files are in a different directory
import sys
sys.path.insert(0, '../data+wrangling')
from util import make_test_train


def best_n(X_train, X_test,y_train,y_test,min_n = 1, max_n = 19):
    '''
        Estimate the best n
    '''
    f1_list = []
    for i in range(min_n,max_n+1):
        nbrs = NearestNeighbors(n_neighbors=i,algorithm='ball_tree').fit(X_train)
        distances, indices = nbrs.kneighbors(X_test)
        pred = []
        for ind in indices:
            y_list = y_train[ind]
            pred.append(np.bincount(y_list).argmax())
        f1_list.append(f1_score(y_test,pred,average="weighted"))
    
    print f1_list[np.argmax(f1_list)]
    return range(min_n,max_n+1)[np.argmax(f1_list)]

def debugger(X_train,X_test,y_train,y_test,n_neighbors=19):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train,y_train)
    pred = neigh.predict(X_test)
    print f1_score(y_test,pred,average="weighted")
    print 'confusion matrix',confusion_matrix(y_test,pred)

def main():
    X_train,X_test,y_train,y_test = make_test_train()
    X_train[:,-12:]
    X_test[:,-12:]
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')
    best =  best_n(X_train,X_test,y_train,y_test,max_n = 100)
    print best
    debugger(X_train,X_test,y_train,y_test,n_neighbors=best)
    #nbrs  = NearestNeghbors(n_neighbors = )

if __name__ == '__main__':
    main()