#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:44:19 2017

@author: DavidVanDusen
"""

import numpy as np
from scipy.io import wavfile
from temp import bufferSig
import matplotlib.pyplot as plt




fs, song = wavfile.read(filepaths[0][5])
song = song/np.max(np.abs(song))

lenInSecs = len(song)/fs
t = np.linspace(0,lenInSecs,len(song))
#plt.plot(t,song)
win_size = 2048
hop_size = 256
min_lag = 15
max_lag = 800
overlap = win_size - hop_size
frames = bufferSig(song,win_size,overlap)

pitch = np.zeros(frames.shape[1])
yin = np.zeros(max_lag)
norms = np.zeros(frames.shape[1])
normalization = 1

for i in range(0,frames.shape[1]):
     for j in range(0,max_lag):
         if j==0:
             yin[j] = 1
         else:
            secLen = win_size-j
            thisDiff = frames[0:secLen,i]-frames[j:win_size,i]
            yin[j] = np.sum(thisDiff*thisDiff)
            normalization = np.sum(yin[0:j])/j;
            yin[j] = yin[j]/(normalization)
     lags = yin[min_lag:max_lag]
     min_index = (np.argmin(yin[min_lag:max_lag]))
     pitch[i] = fs/(min_index+min_lag)



plt.figure()
plt.plot(pitch, 'ko')
plt.ylim([0,500])
