#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:04:50 2017

@author: DavidVanDusen
"""
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
from temp import *

fs, song = wavfile.read('ABirdTest.wav')
print('CMON')
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
thing = filtfilt(b,a,le)
#plt.plot(thing)
thresh = createThreshold(le, 25)

#plt.plot(thresh)
listenTo = le-thresh
listenTo = (listenTo[:]+np.abs(listenTo[:]))/2
listenTo = medfilt(listenTo, 21)
#plt.figure
#plt.plot(listenTo)

[S,F,T]=plotSpectrogram('ABirdTest.wav',win_size,hop_size)
print('What up')
plt.figure
plt.pcolormesh(T,F,S)