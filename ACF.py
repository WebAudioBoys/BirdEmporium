#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:46:51 2017

@author: DavidVanDusen
"""

import numpy as np
from scipy.io import wavfile
from temp import bufferSig
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt

#import scikits.audiolab

fs, song = wavfile.read(filepaths[0][10])
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
#all_lags = int(np.linspace(1,max_lag, max_lag))
pitch = np.zeros(frames.shape[1])

