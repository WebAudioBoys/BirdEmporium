#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:49:13 2017

@author: DavidVanDusen
"""

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(allFrames.transpose(), labels, test_size=0.33, random_state=42)