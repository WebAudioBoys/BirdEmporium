#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:36:47 2017

@author: DavidVanDusen
"""

from MFCCs import compute_mfccs_frames
import numpy as np

allFrames = np.load('TruncatedBirdFrames32.npy')
labels = np.load('BirdLabels32.npy')

min_freq = 500
max_freq = 10000
num_mel_filts = 26
n_dct = 15
win_size = 512
hop_size = 256
fs = 32000


mfccs, mfcc_fs = compute_mfccs_frames(allFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

#num_frames_per_class = np.bincount(labels.astype(int))
#num_frames_per_test = np.floor(np.min(num_frames_per_class)/5)


    
    

