#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:36:47 2017

@author: DavidVanDusen
"""

from MFCCs import compute_mfccs_frames
import numpy as np

orioleFrames = np.load('../SavedVariables/orioleTruncatedFrames32.npy')
cardinalFrames = np.load('../SavedVariables/cardinalTruncatedFrames32.npy')
chickadeeFrames = np.load('../SavedVariables/chickadeeTruncatedFrames32.npy')
finchFrames = np.load('../SavedVariables/finchTruncatedFrames32.npy')
robinFrames = np.load('../SavedVariables/robinTruncatedFrames32.npy')
labels = np.load('../SavedVariables/BirdLabels32.npy')

min_freq = 500
max_freq = 10000
num_mel_filts = 26
n_dct = 15
win_size = 512
hop_size = 256
fs = 32000

print("Oriole MFCCs calculating.")
oriole_mfccs, mfcc_fs = compute_mfccs_frames(orioleFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

print("Cardinal MFCCs calculating.")
cardinal_mfccs, mfcc_fs = compute_mfccs_frames(cardinalFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

print("Chickadee MFCCs calculating.")
chickadee_mfccs, mfcc_fs = compute_mfccs_frames(chickadeeFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

print("Goldfinch MFCCs calculating.")
finch_mfccs, mfcc_fs = compute_mfccs_frames(finchFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

print("Robin MFCCs calculating.")
robin_mfccs, mfcc_fs = compute_mfccs_frames(robinFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

print("Saving results.")
all_mfccs = np.hstack((oriole_mfccs, cardinal_mfccs, chickadee_mfccs, finch_mfccs, robin_mfccs))
np.save('../SavedVariables/all_mfccs', all_mfccs)
#num_frames_per_class = np.bincount(labels.astype(int))
#num_frames_per_test = np.floor(np.min(num_frames_per_class)/5)


    
    

