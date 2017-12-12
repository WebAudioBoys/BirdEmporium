#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:48:08 2017

@author: DavidVanDusen
"""

import numpy as np


def bufferSig(array,win_size,overlap):
    array = array/max(array)
    hopSize = win_size - overlap
    zeroPad = np.zeros(overlap)
    sig = np.append(zeroPad,array)
    currLen = len(sig)
    endZeroPad = np.zeros(((win_size - (currLen%win_size))+win_size))
    sig = np.append(sig,endZeroPad)
    window = np.hanning(win_size)
    numFrames = int(1+(len(array)/hopSize))
    bufferedSig = np.zeros([win_size, numFrames])
    
    for i in range(0,numFrames):
        startIndex = i * hopSize
        endIndex = startIndex+win_size
        bufferedSig[:,i] =  sig[startIndex:endIndex] * window
    return bufferedSig

def getSpectrogram(array,win_size,hop_size,fs):
    #Read in file
    lenInTime = len(array)/fs
    #Length of spectrogram window
    specLen = int(1+(win_size/2))
    #Sample overlap
    overlap = win_size - hop_size
    #Break the signal into frames
    buf=bufferSig(array, win_size,overlap)
    #Take the fft of every frame
    buf = np.fft.fft(buf,win_size,0)
    #Cut them down to size
    buf = buf[0:specLen,:]
    buf = np.abs(buf)
#    buf = 20*np.log10(buf)
    
    F = np.linspace(0,fs/2,specLen)
    T = np.linspace(0,lenInTime,buf.shape[1])
    return  buf,F,T


#S,F,T = getSpectrogram(song, 1024,512,fs)