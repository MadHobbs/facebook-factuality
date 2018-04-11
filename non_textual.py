
import util
from soybeans import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


def main():
    ## Split into train/test
    # TODO: in the fufture, we should fold them later
    X,y = util.load_reaction_counts('merged.csv')
    enum_dic = {'no factual content':0, 'mostly true':1, 'mostly false':2, 'mixture of true and false':3}
    for i in range(len(y)):
        y[i] = enum_dic[y[i]]
    sss = StratifiedShuffleSplit(n_splits=1,test_size =0.3,random_state=0)
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]



    ################################################################
    ## Predict factuality from number of                          ##
    ## shares, comments, likes, loves, wows,hahas, sads, angrys   ##
    ################################################################
    # -- This is about if people respond to posts depending on    ## 
    # -- the factuality of the post. (this might be some          ##
    # -- underlying cusation of media intention)                  ##
    ################################################################
    num_classes = 4
    loss_func_list = [hamming_losses,sigmoid_losses,logistic_losses]
    R_list = [generate_output_codes(num_classes, 'ova'),generate_output_codes(num_classes, 'ovo')]
    code_itr = iter(['ova','ovo']*3)

    print 'classifying...'
    for loss_func in loss_func_list :
        for R in R_list : 
    #   train a multiclass SVM on training data and evaluate on test data
    #   setup the binary classifiers using the specified parameters from the handout
            # clf = MulticlassSVM(R = R, kernel='poly', degree = 4, coef0 = 1, gamma = 1.0)
            clf = MulticlassSVM(R = R, kernel='linear')
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test, loss_func=loss_func)
            print str(loss_func)
            print code_itr.next()
            num_errors = sum(pred != y_test)
            print "number of erros: " + str(num_errors) 
            print 'Accuracy: ', (1.0 - (num_errors/float(len(y))))
            print '\n\n'

######################################################################
## __main__                                                         ##
######################################################################


if __name__ == "__main__" :
    main()