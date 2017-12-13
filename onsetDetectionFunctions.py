#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:36:03 2017

@author: DavidVanDusen
"""
import numpy as np
from basicFunctions import bufferSig,getSpectrogram
from scipy.stats.mstats import gmean
from scipy.signal import medfilt, butter, filtfilt


def plotSpectrogram(array,win_size,hop_size,fs):
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
    buf = 20*np.log10(buf, where=True)
    F = np.linspace(0,fs/2,specLen)
    T = np.linspace(0,lenInTime,buf.shape[1])
    return  buf,F,T

def localEnergy(array,win_size,hop_size,fs):
    overlap = win_size - hop_size
    buf = bufferSig(array,win_size,overlap)

    localEnergies = np.zeros(buf.shape[1])
    for i in range(0,buf.shape[1]):
        localEnergies[i] = np.sum(np.square(buf[:,i]))
    
    for i in range(0,len(localEnergies)):
        if localEnergies[i] > 0:
            localEnergies[i] = np.log10(localEnergies[i])
#    localEnergies = np.log10(localEnergies, where=trueSpot)
    localEnergies = np.diff(localEnergies)
    localEnergies = np.append(np.mean(localEnergies),localEnergies)
    localEnergies = localEnergies-np.min(localEnergies)
    localEnergies = localEnergies/np.max(localEnergies)
    le_fs = fs/hop_size
    return localEnergies, le_fs

def spectralFlux(array,win_size,hop_size,fs):
    spec,F,T = getSpectrogram(array,win_size,hop_size,fs)
    specFlux = np.diff(spec)
    
    specFlux = 0.5*(specFlux+np.abs(specFlux))
    specFluxVals = np.sum(specFlux,axis=0)
    specFluxVals = specFluxVals/specFluxVals.shape[0] 
    sf_fs = fs/hop_size
    return specFluxVals,sf_fs       

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

def noveltyLPF(nov, fs, w_c):
    w_c = 2*w_c/fs
    [b, a] = butter(3,w_c,btype='low')
    filtered_le = filtfilt(b,a,nov)
    return filtered_le

def createThreshold(array,filtLen):
    threshold = medfilt(array, filtLen)
    return threshold

def spectralFlatness(spec):
    numerator = gmean(spec,axis=0)
    denom = np.mean(spec,axis=0)
    output = numerator/denom
    return output

def threshPeaks(le,thresh):
    le_diff = np.diff(le) 
    #Returns indices of zero cross in diff
    le_diff = np.append(0,le_diff)
    #signal = np.sign(signal)
    diffZC = np.diff(np.sign(le_diff))
         #Only find upper peaks
    diffZC = -1 * diffZC
    diffZC = (diffZC + np.abs(diffZC))/2
    output = np.where(diffZC)
    
    output = np.asarray(output) + 1
#    output = np.asarray(output)
    values = le[output[:]]
    threshAtPeaks = thresh[output[:]]
    peak_diff = values-threshAtPeaks
    threshAtPeaks = peak_diff + np.abs(peak_diff)
    properPeaks = np.where(threshAtPeaks)
    peaks = values[properPeaks[:]]
    times = output[properPeaks[:]]
    return peaks, times
    
    