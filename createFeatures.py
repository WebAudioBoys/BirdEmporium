#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:36:47 2017

@author: DavidVanDusen
"""

from MFCCs import compute_mfccs_frames,dft_of_frames,compute_mfccs_from_spec,filter_noisy_frames
import numpy as np
from onsetDetectionFunctions import spectralFlatness

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


print("Filtering noisy frames")
orioleFrames, num_deleted_frames = filter_noisy_frames(orioleFrames)
print("Oriole MFCCs calculating.")
oriole_mfccs, mfcc_fs = compute_mfccs_frames(orioleFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
oriole_len = oriole_mfccs.shape[1]

print("Filtering noisy frames")
cardinalFrames, num_deleted_frames = filter_noisy_frames(cardinalFrames)
print("Cardinal MFCCs calculating.")
cardinal_mfccs, mfcc_fs = compute_mfccs_frames(cardinalFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
cardinal_len = cardinal_mfccs.shape[1]

print("Filtering noisy frames")
chickadeeFrames, num_deleted_frames = filter_noisy_frames(chickadeeFrames)
print("Chickadee MFCCs calculating.")
chickadee_mfccs, mfcc_fs = compute_mfccs_frames(chickadeeFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
chickadee_len = chickadee_mfccs.shape[1]

print("Filtering noisy frames")
finchFrames, num_deleted_frames = filter_noisy_frames(finchFrames)
print("Goldfinch MFCCs calculating.")
finch_mfccs, mfcc_fs = compute_mfccs_frames(finchFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
finch_len = finch_mfccs.shape[1]

print("Filtering noisy frames")
robinFrames, num_deleted_frames = filter_noisy_frames(robinFrames)
print("Robin MFCCs calculating.")
robin_mfccs, mfcc_fs = compute_mfccs_frames(robinFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
robin_len = robin_mfccs.shape[1]

print("Saving results.")
all_mfccs = np.hstack((oriole_mfccs, cardinal_mfccs, chickadee_mfccs, finch_mfccs, robin_mfccs))
labels = np.hstack((np.zeros(oriole_len),np.zeros(cardinal_len)+1,np.zeros(chickadee_len)+2,np.zeros(finch_len)+3,np.zeros(robin_len)+4))

#np.save('../SavedVariables/all_mfccs', all_mfccs)
#num_frames_per_class = np.bincount(labels.astype(int))
#num_frames_per_test = np.floor(np.min(num_frames_per_class)/5)


    
    

