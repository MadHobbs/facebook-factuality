import numpy as np
import matplotlib.pyplot as plt
import validations
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score, confusion_matrix
from sklearn.dummy import DummyClassifier
# modify path because these files are in a different directory
import sys
sys.path.insert(0, '../data+wrangling')
import util

X_train, X_test, y_train, y_test = util.make_test_train()
clf = DummyClassifier(strategy = "most_frequent")
clf.fit(X_train, y_train)

preds = clf.predict(X_train)
print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) 
preds = clf.predict(X_test)
print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) 
print "confusion matrix trained with f1: \n" + str(confusion_matrix(y_test, preds))

preds = clf.predict(X_train)
print "train accuracy: " + str(accuracy_score(y_train, preds))
preds = clf.predict(X_test)
print "test accuracy: " + str(accuracy_score(y_test, preds))
print "confusion matrix trained with accuracy: \n" + str(confusion_matrix(y_test, preds)) 