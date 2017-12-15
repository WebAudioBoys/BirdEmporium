# alternate grabGoodFrames version
# individual arrays for species, possibly improved runtime

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
orioleFrames = np.zeros([win_size,1])
cardinalFrames = np.zeros([win_size,1])
chickadeeFrames = np.zeros([win_size,1])
finchFrames = np.zeros([win_size,1])
robinFrames = np.zeros([win_size,1])
labels = np.array([])
w_c = 1.8

path = '../32kHzBirdSongs/'
filepaths = grabFilepaths(path)

# for each species directory
for directory in range(0,5):
    # for each file in the species directory
    for filepath in filepaths[directory]:
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
        max_num_frames = np.floor(fs/hop_size)
        endTimes = np.append(np.diff(times),0)
        endTimes = np.clip(endTimes,1,max_num_frames)
        endTimes = times+endTimes
        endTimes = np.asarray(endTimes, dtype=np.int64)
        lengths = endTimes-times
        numFrames = np.sum(endTimes-times)
        framesWeNeed = np.zeros([frames.shape[0],numFrames])

        # extract relevant frames, append to output
        startIndex = 0
        if len(lengths) != 0:
            endIndex = lengths[0]
            for i in range(0,len(times)-1):  
                framesWeNeed[:,startIndex:endIndex] = frames[:,times[i]:endTimes[i]]
                startIndex = startIndex + lengths[i]
                endIndex = startIndex + lengths[i+1]
            if directory == 0:
                orioleFrames = np.hstack((orioleFrames,framesWeNeed))
            if directory == 1:
                cardinalFrames = np.hstack((cardinalFrames,framesWeNeed))
            if directory == 2:
                chickadeeFrames = np.hstack((chickadeeFrames,framesWeNeed))
            if directory == 3:
                finchFrames = np.hstack((finchFrames,framesWeNeed))
            if directory == 4:
                robinFrames = np.hstack((robinFrames,framesWeNeed))
        print(filepath, 'is finished buddy!')

        # create labels
        theseLabels = np.ones(framesWeNeed.shape[1])
        labels = np.append(labels, theseLabels*directory)

# save outputs as .npz for feature extraction
orioleFrames = orioleFrames[:,1:]      
cardinalFrames = cardinalFrames[:,1:]      
chickadeeFrames = chickadeeFrames[:,1:]      
finchFrames = finchFrames[:,1:]      
robinFrames = robinFrames[:,1:]      

np.save('../SavedVariables/orioleTruncatedFrames32', orioleFrames)
np.save('../SavedVariables/cardinalTruncatedFrames32', cardinalFrames)
np.save('../SavedVariables/chickadeeTruncatedFrames32', chickadeeFrames)
np.save('../SavedVariables/finchTruncatedFrames32', finchFrames)
np.save('../SavedVariables/robinTruncatedFrames32', robinFrames)

np.save('../SavedVariables/BirdLabels32', labels)        
