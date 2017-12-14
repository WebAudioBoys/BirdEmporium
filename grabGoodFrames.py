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
from onsetDetectionFunctions import noveltyLPF,spectralFlux,localEnergy,findPeaks,createThreshold,threshPeaks

win_size = 512
hop_size = 256
overlap = win_size - hop_size
allFrames = np.zeros([win_size,1])
labels = np.array([])
w_c = 1.8

path = '../32kHzBirdSongs/'
filepaths = grabFilepaths(path)
for directory in range(0,5):
    for filepath in filepaths[directory]:
        fs, song = wavfile.read(filepath)
        if len(song) > fs*60:
            song = song[0:fs*60]
        song = song/np.max(np.abs(song))
        frames = bufferSig(song,win_size,overlap)
#        le, le_fs = localEnergy(song,win_size,hop_size,fs)
#        w_c = 4
#        filtered_le = noveltyLPF(le,le_fs,w_c)
#        peaks = findPeaks(filtered_le)
#        thresh = createThreshold(filtered_le,9)
#        peaks,times = threshPeaks(filtered_le,thresh)
#        max_num_frames = np.floor(fs/hop_size)
##        test the below concept in the termy
#        endTimes = np.append(np.diff(times),0)
#        endTimes = np.clip(endTimes,1,max_num_frames)
#        endTimes = times+endTimes
#        endTimes = np.asarray(endTimes, dtype=np.int64)
#        lengths = endTimes-times
#        numFrames = np.sum(endTimes-times)
#        framesWeNeed = np.zeros([frames.shape[0],numFrames])
        
        
        
        
        sf,sf_fs = spectralFlux(song,win_size,hop_size,fs)
        sf = np.append(0,sf)
        filtered_sf = noveltyLPF(sf,sf_fs,w_c)
        filtered_sf = (filtered_sf - np.min(filtered_sf))
        filtered_sf = filtered_sf/np.max(filtered_sf)
        thresh2 = createThreshold(filtered_sf,31)
        thresh2 = thresh2 + 0.01
        peaks,times = threshPeaks(filtered_sf,thresh2)
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
        if len(lengths) != 0:
            endIndex = lengths[0]
            
            for i in range(0,len(times)-1):
                
                framesWeNeed[:,startIndex:endIndex] = frames[:,times[i]:endTimes[i]]
                startIndex = startIndex + lengths[i]
                endIndex = startIndex + lengths[i+1]
            allFrames = np.hstack((allFrames,framesWeNeed))
        print(filepath, 'is finished buddy!')
        theseLabels = np.ones(framesWeNeed.shape[1])
        labels = np.append(labels, theseLabels*directory)

allFrames = allFrames[:,1:]      
np.save('BirdLabels32', labels)        
np.save('TruncatedBirdFrames32', allFrames)
        
        