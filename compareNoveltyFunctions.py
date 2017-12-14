#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:00:25 2017

@author: DavidVanDusen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from grabFilePaths import grabFilepaths
from basicFunctions import bufferSig
from onsetDetectionFunctions import spectralFlux,noveltyLPF,localEnergy,createThreshold,threshPeaks
from scipy.signal import medfilt

path = '../32kHzBirdSongs/'
filepaths = grabFilepaths(path)
win_size = 512
hop_size = 256
overlap = win_size - hop_size
fs, song = wavfile.read(filepaths[3][4])
if len(song) > fs*60:
    song = song[0:fs*60]
song = song/np.max(np.abs(song))
songLen = len(song)/fs
frames = bufferSig(song,win_size,overlap)
timeBins = frames.shape[1]
timeVec = np.linspace(0,1-(1/timeBins), timeBins)
timeVec = timeVec * songLen
le, le_fs = localEnergy(song,win_size,hop_size,fs)
w_c = 1.8
filtered_le = noveltyLPF(le,le_fs,w_c)
filtered_le = (filtered_le - np.min(filtered_le))
filtered_le = filtered_le/np.max(filtered_le)
thresh1 = createThreshold(filtered_le,15)
thresh1 = thresh1 + 0.003
peaks,times = threshPeaks(filtered_le,thresh1)

plt.figure()
plt.subplot(211)
plt.plot(timeVec,filtered_le,'r')
plt.plot(timeVec,thresh1,':b')
plt.plot(timeVec[times[:]],peaks, 'ko')
plt.xlabel('Time in Seconds')
plt.ylabel('Nov Broiii')

sf,sf_fs = spectralFlux(song,win_size,hop_size,fs)
sf = np.append(0,sf)
filtered_sf = noveltyLPF(sf,sf_fs,w_c)
filtered_sf = (filtered_sf - np.min(filtered_sf))
filtered_sf = filtered_sf/np.max(filtered_sf)
thresh2 = createThreshold(filtered_sf,31)
thresh2 = thresh2 + 0.01
peaks,times = threshPeaks(filtered_sf,thresh2)


plt.subplot(212)
plt.plot(timeVec,filtered_sf,'r')
plt.plot(timeVec,thresh2,':b')
plt.plot(timeVec[times[:]],peaks, 'ko')
plt.xlabel('Time in Seconds')
plt.ylabel('Nov Broiii')

#plt.subplot(212)
#plt.plot()
