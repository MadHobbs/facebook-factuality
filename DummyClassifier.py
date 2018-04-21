import util
import numpy as np
import matplotlib.pyplot as plt
import validations
from sklearn.metrics import f1_score, make_scorer, accuracy_score, average_precision_score, confusion_matrix
from sklearn.dummy import DummyClassifier


X_train, X_test, y_train, y_test, colnames = util.make_test_train()
clf = DummyClassifier(strategy = "most_frequent")
clf.fit(X_train, y_train)

preds = clf.predict(X_train)
print "train f1_score: " + str(f1_score(y_train, preds, average="weighted")) # 0.9958
preds = clf.predict(X_test)
print "test f1_score: " + str(f1_score(y_test, preds, average="weighted")) # 0.88877
print "confusion matrix trained with f1: \n" + str(confusion_matrix(y_test, preds))

preds = clf.predict(X_train)
print "train accuracy: " + str(accuracy_score(y_train, preds)) # 1.0
preds = clf.predict(X_test)
print "test accuracy: " + str(accuracy_score(y_test, preds))
print "confusion matrix trained with accuracy: \n" + str(confusion_matrix(y_test, preds)) #0.887