#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:22:21 2017

@author: DavidVanDusen
"""

###This script reads in all of the filepaths in the enclosing directory
path = '../BirdSongs/'
import os
import numpy as np
filepaths = []
enclosingFolder = os.listdir(path)
enclosingFolder.pop(0)
for folder in enclosingFolder:
    currPath = path+folder+'/'
    thisSet = os.listdir(path+folder)
    if thisSet[0][0] == '.':
        thisSet.pop(0)
    thisSet = [currPath+s for s in thisSet]
    filepaths.append(thisSet)
    