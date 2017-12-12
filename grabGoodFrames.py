#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:46:13 2017

@author: DavidVanDusen
"""

import os
from grabFilePaths import *
from basicFunctions import *

win_size = 128
hop_size = 64
overlap = win_size - hop_size

path = '../16kHzBirdSongs/'
filepaths = grabFilepaths(path)
for directory in range(0,5):
    for filepath in filepaths[directory]: 
        frames = bufferSig(array,win_size,overlap)
        le, le_fs = localEnergy(array,win_size,hop_size,fs)
        w_c = 4
        noveltyLPF(le,le_fs, 4)
        peaks = findPeaks(filtered_le)
        thresh = createThreshold(le,9)
        peaks,times = threshPeaks(le,thresh)
        num_frames