#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:36:03 2017

@author: DavidVanDusen
"""
import numpy as np
from basicFunctions import *
from scipy.stats.mstats import gmean


def plotSpectrogram(array,win_size,hop_size):
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
    buf = np.abs(buf[0:specLen,:])
    buf = 20*np.log10(buf)
    F = np.linspace(0,fs/2,specLen)
    T = np.linspace(0,lenInTime,buf.shape[1])
    return  buf,F,T

def localEnergy(array,win_size,hop_size,fs):
    buf = bufferSig(array,win_size,hop_size)
    localEnergies = np.zeros(buf.shape[1])
    for i in range(0,buf.shape[1]):
        localEnergies[i] = (np.sum(np.square(buf[:,i])))/win_size
    localEnergies = np.log10(localEnergies)
    localEnergies = np.diff(localEnergies)
    localEnergies = np.append(0,localEnergies)
    localEnergies = localEnergies-np.min(localEnergies)
    localEnergies = localEnergies/np.max(localEnergies)
    le_fs = fs/hop_size
    return localEnergies, le_fs

def spectralFlux(array,win_size,hop_size,fs):
    spec,F,T = getSpectrogram(array,win_size,hop_size,fs)
    specFlux = np.diff(spec,1,1)
    specFlux = np.append(np.zeros(win_size,1),specFlux)
    specFlux = 0.5*(specFlux+np.abs(specFlux))
    specFlux = np.sum(specFlux,axis=0)/spec.shape[0]        

def findPeaks(signal):
    signal = np.diff(signal) 
    #Returns indices of zero cross in diff
    np.append([0],signal)
    diffZC = np.where((np.diff(np.sign(signal))))[0]
    diffZC+=1
    #Only find upper peaks
    diffZC = -1 * diffZC
    diffZC = (diffZC + np.abs(diffZC))/2

    return diffZC

def createThreshold(array,filtLen):
    threshold = medfilt(array, filtLen)
    
    return threshold

def spectralFlatness(spec):
    numerator = gmean(spec,axis=0)
    denom = np.mean(spec,axis=0)
    output = numerator/denom
    return output
    
    