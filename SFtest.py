#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:24:34 2017

@author: DavidVanDusen
"""

import numpy as np
from basicFunctions import *
from temp import *
from scipy.stats.mstats import gmean
from scipy.io import wavfile
from onsetDetectionFunctions import spectralFlatness
import matplotlib.pyplot as plt
fs, song = wavfile.read(filepaths[1][15])
song = song/np.max(np.abs(song))

lenInSecs = len(song)/fs
t = np.linspace(0,lenInSecs,len(song))

S,F,T=getSpectrogram(song,1024,512,fs)
x = spectralFlatness(S)

xx = np.diff(x)
xx = np.append(0,xx)
xx = xx * -1
x = (x - np.min(x))/np.max(x)
xx = (xx - np.min(xx))/np.max(xx)
plt.figure()
plt.plot(x ,'k')
plt.plot(xx,'b')