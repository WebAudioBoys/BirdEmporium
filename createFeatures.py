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
lengths = np.array([])

print("Oriole MFCCs calculating.")
oriole_mfccs, mfcc_fs = compute_mfccs_frames(orioleFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
lengths = np.append(lengths,oriole_mfccs.shape[1])
print("Cardinal MFCCs calculating.")
cardinal_mfccs, mfcc_fs = compute_mfccs_frames(cardinalFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
lengths = np.append(lengths,cardinal_mfccs.shape[1])
print("Chickadee MFCCs calculating.")
chickadee_mfccs, mfcc_fs = compute_mfccs_frames(chickadeeFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
lengths = np.append(lengths,chickadee_mfccs.shape[1])
print("Goldfinch MFCCs calculating.")
finch_mfccs, mfcc_fs = compute_mfccs_frames(finchFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
lengths = np.append(lengths,finch_mfccs.shape[1])
print("Robin MFCCs calculating.")
robin_mfccs, mfcc_fs = compute_mfccs_frames(robinFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)
lengths = np.append(lengths,robin_mfccs.shape[1])

per_class_len = np.int(np.min(lengths))

orioleIndices = np.random.permutation(np.arange(lengths[0],dtype=np.int))[:per_class_len]
cardinalIndices = np.random.permutation(np.arange(lengths[1],dtype=np.int))[:per_class_len]
chickadeeIndices = np.random.permutation(np.arange(lengths[2],dtype=np.int))[:per_class_len]
finchIndices = np.random.permutation(np.arange(lengths[3],dtype=np.int))[:per_class_len]
robinIndices = np.random.permutation(np.arange(lengths[4],dtype=np.int))[:per_class_len]

oriole_mfccs = oriole_mfccs[:,orioleIndices[:]]
cardinal_mfccs = cardinal_mfccs[:,cardinalIndices[:]]
chickadee_mfccs = chickadee_mfccs[:,chickadeeIndices[:]]
finch_mfccs = finch_mfccs[:,finchIndices[:]]
robin_mfccs = robin_mfccs[:,robinIndices[:]]

labels = np.array([])
print("Saving results.")
for i in range(5):
    class_nums = np.zeros(per_class_len)+ i
    labels = np.append(labels,class_nums)
    

all_mfccs = np.hstack((oriole_mfccs, cardinal_mfccs, chickadee_mfccs, finch_mfccs, robin_mfccs))

np.save('../SavedVariables/all_mfccs', all_mfccs)
#num_frames_per_class = np.bincount(labels.astype(int))
#num_frames_per_test = np.floor(np.min(num_frames_per_class)/5)


    
    

