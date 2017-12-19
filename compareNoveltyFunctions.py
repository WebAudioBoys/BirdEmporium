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
from onsetDetectionFunctions import spectralFlux,noveltyLPF,createThreshold,threshPeaks


path = '../32kHzBirdSongs/'
filepaths = grabFilepaths(path)
win_size = 512
hop_size = 256
w_c = 1.8
overlap = win_size - hop_size
fs, song = wavfile.read(filepaths[0][10])

if len(song) > fs*60:
    song = song[0:fs*60]
songLen = len(song)
songTime = np.linspace(0,len(song)-1,songLen)
songTime = songTime/fs
song = song/np.max(np.abs(song))

frames = bufferSig(song,win_size,overlap)
timeBins = frames.shape[1]
timeVec = np.linspace(0,1-(1/timeBins), timeBins)
timeVec = timeVec * (songLen/fs)

plt.figure()
plt.subplot(211)
plt.plot(songTime,song,'r')
plt.xlabel('Time in Seconds')
plt.ylabel('Amplitude')
plt.title('Baltimore Oriole')

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
plt.ylabel('Spectral Flux')

