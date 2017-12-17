#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:04:22 2017

@author: bgowland
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
mfccs = np.load('../SavedVariables/all_mfccs.npy')
labels = np.load('../SavedVariables/BirdLabels32.npy') 
birds = ["Oriole", "Cardinal", "Chickadee", "Goldfinch", "Robin"]

print("Splitting testing and training data.")
X_train, X_test, y_train, y_test = train_test_split(mfccs.transpose(), labels, test_size=0.20, random_state=42)
print("Preparing random forest.")
forest = RandomForestClassifier(n_estimators=100, random_state=0, max_features=4)
print("Classifying.")
forest.fit(X_train, y_train)

print("--------------------")
print("Results report")
print("--------------------")

print("Overall accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Overall accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
print("--------------------")

# precision, recall, f measure stats
p_labels = forest.predict(X_test)

class_precision = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
class_recall = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
class_f_measure = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

for i in range(0,5):
    c = len(p_labels[(p_labels==y_test) & (p_labels==i)])
    f_p = len(p_labels[(p_labels!=y_test) & (p_labels==i)])
    f_n = len(p_labels[(p_labels!=y_test) & (y_test==i)])
    class_precision[i] = c/(c + f_p)
    class_recall[i] = c/(c + f_n)
    class_f_measure[i] = (2 * class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
    print("%s results:" % birds[i])
    print("Precision: %f" % class_precision[i])
    print("Recall: %f" % class_recall[i])
    print("F Measure: %f" % class_f_measure[i])
    print("--------------------")

avg_precision = np.average(class_precision)
avg_recall = np.average(class_recall)
avg_f_measure = np.average(class_f_measure)

print("Average precision: %f" % avg_precision)
print("Average recall: %f" % avg_recall)
print("Average F measure: %f" % avg_f_measure)
