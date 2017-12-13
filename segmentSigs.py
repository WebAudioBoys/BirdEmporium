#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:56:53 2017

@author: DavidVanDusen
"""

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from temp import *
#import scikits.audiolab

fs, song = wavfile.read(filepaths[2][30])
song = song/np.max(np.abs(song))

lenInSecs = len(song)/fs
t = np.linspace(0,lenInSecs,len(song))
#plt.plot(t,song)
win_size = 1024
hop_size = 512
overlap = win_size - hop_size
frames = bufferSig(song,win_size,overlap)
le, le_fs = localEnergy(song,win_size, hop_size, fs)

w_c = 4 #in Hz
w_c = 2*w_c/le_fs
[b, a] = butter(1,w_c,btype='lowpass')
filtered_le = filtfilt(b,a,le)
peaks = findPeaks(filtered_le)
thresh = createThreshold(le, 25)
thresh = thresh + 0.06
plt.plot(thresh, 'b')


signal = np.diff(filtered_le) 
    #Returns indices of zero cross in diff
signal = np.append(0,signal)
#signal = np.sign(signal)
diffZC = np.diff(np.sign(signal))
     #Only find upper peaks
diffZC = -1 * diffZC
diffZC = (diffZC + np.abs(diffZC))/2
output = np.where(diffZC)
output = np.asarray(output)
times =output[:] 
values = filtered_le[output[:]]

threshAtPeaks = thresh[times[:]]
threshAtPeaks = filtered_le[times[:]]-threshAtPeaks
threshAtPeaks = threshAtPeaks + np.abs(threshAtPeaks)
properPeaks = np.where(threshAtPeaks)
peaks = values[properPeaks[:]]
times = times[properPeaks[:]]

#scikits.audiolab.play(song,fs)    
plt.figure()
plt.plot(thresh, 'b')
plt.plot(filtered_le, 'r')
plt.plot(times,peaks, 'ko')

segLen = np.int_((np.median(np.diff(times))))
endTimes = times + segLen
#plt.ylim((0,1))
    
#    for i in range(0,len(diffZC)):
#        if signal[diffZC[i]] > thresh:
#            np.append(output,signal[diffZC[i]])
#        else:
#            
numFrames = np.sum(endTimes-times)
framesWeNeed = np.zeros([frames.shape[0],numFrames])
for i in range(0,len(times)):
    startIndex = i*segLen
    endIndex = startIndex+segLen
    framesWeNeed[:,startIndex:endIndex] = frames[:,times[i]:endTimes[i]]

min_freq = 500
max_freq = 4000
num_mel_filts = 40
n_dct = 15    
#
#mfccs, mfcc_fs=compute_mfccs_frames(frames,fs,hop_size,min_freq,max_freq, 
#                                    num_mel_filts, n_dct)
#
#plt.pcolormesh(mfccs)