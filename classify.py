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

print("Splitting testing and training data.")
X_train, X_test, y_train, y_test = train_test_split(mfccs.transpose(), labels, test_size=0.20, random_state=42)
print("Preparing random forest.")
forest = RandomForestClassifier(n_estimators=200, random_state=0, max_features=4)
print("Classifying.")
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))