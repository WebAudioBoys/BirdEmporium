#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:04:20 2017

@author: bgowland
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.io import wavfile
from scipy import stats
from basicFunctions import bufferSig
from onsetDetectionFunctions import noveltyLPF,spectralFlux,createThreshold,threshPeaks
from MFCCs import compute_mfccs_frames

# change filepath and species # for each test
filepath = '../DemoBirds/chickadee.wav'
species = 3
pretty_print = ['an oriole!', 'a cardinal!', 'a chickadee!', 'a goldfinch!', 'a robin!']

# load training data and prepare random forest
X_train = np.load('../SavedVariables/X_train.npy')
y_train = np.load('../SavedVariables/y_train.npy')

# comment this out if forest is ready to go
#print("Preparing random forest.")
#forest = RandomForestClassifier(n_estimators=100, random_state=0, max_features=4)
#forest.fit(X_train, y_train)

# load and prepare test data
win_size = 512
hop_size = 256
overlap = win_size - hop_size
orioleFrames = np.zeros([win_size,1])
cardinalFrames = np.zeros([win_size,1])
chickadeeFrames = np.zeros([win_size,1])
finchFrames = np.zeros([win_size,1])
robinFrames = np.zeros([win_size,1])
labels = np.array([])
w_c = 1.8

# read and buffer, truncate to 60s for long signals
fs, song = wavfile.read(filepath)
if len(song) > fs*60:
    song = song[0:fs*60]
song = song/np.max(np.abs(song))
frames = bufferSig(song,win_size,overlap)

# calculate onsets by spectral flux
sf,sf_fs = spectralFlux(song,win_size,hop_size,fs)
sf = np.append(0,sf)

# smooth and threshold
filtered_sf = noveltyLPF(sf,sf_fs,w_c)
filtered_sf = (filtered_sf - np.min(filtered_sf))
filtered_sf = filtered_sf/np.max(filtered_sf)
thresh2 = createThreshold(filtered_sf,31)
thresh2 = thresh2 + 0.01

# pick peaks
peaks,times = threshPeaks(filtered_sf,thresh2)

# choose frames from onset points
max_num_frames = np.floor(fs/(2*hop_size))
endTimes = np.append(np.diff(times),0)
endTimes = np.clip(endTimes,1,max_num_frames)
endTimes = times+endTimes
endTimes = np.asarray(endTimes, dtype=np.int64)
lengths = endTimes-times
numFrames = np.sum(endTimes-times)
testFrames = np.zeros([frames.shape[0],numFrames])

# extract relevant frames, append to output
startIndex = 0
if len(lengths) != 0:
    endIndex = lengths[0]
    for i in range(0,len(times)-1):  
        testFrames[:,startIndex:endIndex] = frames[:,times[i]:endTimes[i]]
        startIndex = startIndex + lengths[i]
        endIndex = startIndex + lengths[i+1]

print(filepath, ' is finished buddy!')

# create labels
y_test = np.ones(testFrames.shape[1]) * species

# create features
min_freq = 500
max_freq = 10000
num_mel_filts = 26
n_dct = 15
win_size = 512
hop_size = 256
fs = 32000

X_test, mfcc_fs = compute_mfccs_frames(testFrames,fs,
hop_size,min_freq,max_freq,num_mel_filts, n_dct)

# predict class of test
p_labels = forest.predict(X_test.transpose())
p_species = int(stats.mode(p_labels)[0][0])

print("I think this bird is %s" % pretty_print[p_species])
print("Am I right?")
