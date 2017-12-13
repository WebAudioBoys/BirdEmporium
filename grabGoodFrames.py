#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:46:13 2017

@author: DavidVanDusen
"""


import numpy as np
from scipy.io import wavfile
from grabFilePaths import grabFilepaths
from basicFunctions import bufferSig
from onsetDetectionFunctions import noveltyLPF,localEnergy,findPeaks,createThreshold,threshPeaks

win_size = 128
hop_size = 64
overlap = win_size - hop_size
allFrames = np.zeros([win_size,1])

path = '../16kHzBirdSongs/'
filepaths = grabFilepaths(path)
for directory in range(0,5):
    for filepath in filepaths[directory]:
        fs, song = wavfile.read(filepath)
#        if len(song) > fs*60:
#            song = song[0:fs*60]
        song = song/np.max(np.abs(song))
        frames = bufferSig(song,win_size,overlap)
        le, le_fs = localEnergy(song,win_size,hop_size,fs)
        w_c = 4
        filtered_le = noveltyLPF(le,le_fs,w_c)
        peaks = findPeaks(filtered_le)
        thresh = createThreshold(filtered_le,9)
        peaks,times = threshPeaks(filtered_le,thresh)
        max_num_frames = np.floor(fs/hop_size)
#        test the below concept in the termy
        endTimes = np.append(np.diff(times),0)
        endTimes = np.clip(endTimes,1,max_num_frames)
        endTimes = times+endTimes
        endTimes = np.asarray(endTimes, dtype=np.int64)
        lengths = endTimes-times
        numFrames = np.sum(endTimes-times)
        framesWeNeed = np.zeros([frames.shape[0],numFrames])
        startIndex = 0
        endIndex = lengths[0]
        
        for i in range(0,len(times)-1):
            
            framesWeNeed[:,startIndex:endIndex] = frames[:,times[i]:endTimes[i]]
            startIndex = startIndex + lengths[i]
            endIndex = startIndex + lengths[i+1]
        allFrames = np.hstack((allFrames,framesWeNeed))
        print(filepath, 'is finished buddy!')
        
        
        