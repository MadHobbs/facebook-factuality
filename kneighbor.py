from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
import numpy as np
from util import make_test_train


def best_n(X_train, X_test,y_train,y_test,min_n = 1, max_n = 19):
    f1_list = []
    for i in range(min_n,max_n+1):
        nbrs = NearestNeighbors(n_neighbors=i,algorithm='ball_tree').fit(X_train)
        distances, indices = nbrs.kneighbors(X_test)
        pred = []
        for ind in indices:
            y_list = y_train[ind]
            pred.append(np.bincount(y_list).argmax())
        f1_list.append(f1_score(y_test,pred))
    print max_n
    print f1_list[np.argmax(f1_list)]
    return range(min_n,max_n+1)[np.argmax(f1_list)]

def main():
    X_train,X_test,y_train,y_test = make_test_train()
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64') 
    print best_n(X_train,X_test,y_train,y_test,max_n = 55)

    #nbrs  = NearestNeghbors(n_neighbors = )

if __name__ == '__main__':
    main()