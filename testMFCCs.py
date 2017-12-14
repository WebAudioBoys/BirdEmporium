#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:46:46 2017

@author: DavidVanDusen
"""
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from MFCCs import freq2mel,mel2freq
from grabFilePaths import grabFilepaths
from basicFunctions import bufferSig
import matplotlib.pyplot as plt

#def compute_mfccs_frames(frames, fs, hop_size,min_freq,
#                  max_freq, num_mel_filts, n_dct):


min_freq = 500
max_freq = 8000
num_mel_filts = 26
n_dct = 15
win_size = 512
hop_size = 256
overlap = win_size-hop_size


path = '../32kHzBirdSongs/'
filepaths = grabFilepaths(path)
fs, song = wavfile.read(filepaths[2][40])
if len(song) > fs*60:
    song = song[0:fs*60]
song = song/np.max(np.abs(song))
frames = bufferSig(song,win_size,overlap)

win_size = frames.shape[0]
#Length of spectrogram window
specLen = int(1+(win_size/2))
#Sample overlap

#Take the fft of every frame
frames = np.fft.fft(frames,win_size,0)
#Cut them down to size
frames = np.absolute(frames[0:specLen,:])

#Convert to mel
min_freq = freq2mel(min_freq)
max_freq = freq2mel(max_freq)
freqsVec = np.linspace(min_freq,max_freq,num_mel_filts+2)
#    linDiff = freqsVec[1] - freqsVec[0]
#    lowerBound = min_freq - linDiff
#    upperBound = max_freq + linDiff
#    freqsVec = np.append([lowerBound],freqsVec)
#    freqsVec = np.append(freqsVec,[upperBound])
freqsVec = mel2freq(freqsVec[:])
binHop = fs/win_size
closestBins = np.round(freqsVec[:]/binHop)
#this probably does not need to be -1
closestBins = closestBins[:]
print(closestBins)
melFiltBank = np.zeros([num_mel_filts,(frames.shape[0])])
for i in range(0,num_mel_filts):
    startIndex = int(closestBins[i])
    midIndex = int(closestBins[i+1])
    endIndex = int(closestBins[i+2])
    melFiltBank[i,startIndex:midIndex+1] = np.linspace(0,1, midIndex-startIndex+1)
    melFiltBank[i,midIndex:endIndex+1] = np.linspace(1,0, endIndex-midIndex+1)
    melFiltBank[i,midIndex] = 1.0
    #normalize here
    melFiltBank[i,:] = melFiltBank[i,:]/np.sum(melFiltBank[i,:])
    
melFiltered = np.matmul(melFiltBank,frames)
melFiltered = 20*np.log10(melFiltered,where=True)
melFiltered = dct(melFiltered,2,40,0)
#plt.pcolor(melFiltered)
melFiltered = melFiltered[1:n_dct,:]
melFiltered = melFiltered - np.min(melFiltered,axis=0)
melFiltered = melFiltered/np.sum(melFiltered,axis=0) 
melFiltered = np.nan_to_num(melFiltered)
mfcc_fs = hop_size/fs
plt.figure()
plt.pcolor(melFiltered)
####Normalization
#return melFiltered, mfcc_fs
