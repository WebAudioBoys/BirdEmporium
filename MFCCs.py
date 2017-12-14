#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:13:23 2017

@author: DavidVanDusen
"""
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct


#We are going to provide the frames for this function but currently the MFCCs function
#reads in an audio file... Basically all we need is the computation of the filter bank
#I would take care of that now, but I have to go to work...


def mel2freq(melval):
    hzval = 700 * (np.expm1(melval/1127.01028));
    return hzval

def freq2mel(freq):
    melval = 1127.01028 * np.log(1 + (freq/700));
    return melval

def compute_mfccs(file, win_size, hop_size,min_freq,
                  max_freq, num_mel_filts, n_dct):

    #Read in file
    fs, array = wavfile.read(file)
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
    
    #Convert to mel
    min_freq = freq2mel(min_freq)
    max_freq = freq2mel(max_freq)
    freqsVec = np.linspace(min_freq,max_freq,num_mel_filts)
    linDiff = freqsVec[1] - freqsVec[0]
    lowerBound = min_freq - linDiff
    upperBound = max_freq + linDiff
    freqsVec = np.append([lowerBound],freqsVec)
    freqsVec = np.append(freqsVec,[upperBound])
    freqsVec = mel2freq(freqsVec[:])
    binHop = fs/win_size
    closestBins = np.round(freqsVec[:]/binHop)
    #this probably does not need to be -1
    closestBins = closestBins[:]
    melFiltBank = np.zeros([num_mel_filts,(buf.shape[0])])
    for i in range(0,num_mel_filts):
        startIndex = int(closestBins[i])
        midIndex = int(closestBins[i+1])
        endIndex = int(closestBins[i+2])
        melFiltBank[i,startIndex:midIndex+1] = np.linspace(0,1, midIndex-startIndex+1)
        melFiltBank[i,midIndex:endIndex+1] = np.linspace(1,0, endIndex-midIndex+1)
        melFiltBank[i,midIndex] = 1.0
        #normalize here
        melFiltBank[i,:] = melFiltBank[i,:]/np.sum(melFiltBank[i,:])
    print(" Now for the matrix multiplication")    
    melFiltered = np.matmul(melFiltBank,buf)
    melFiltered = 20*np.log10(melFiltered)
    melFiltered = dct(melFiltered,2,40,0)
    #plt.pcolor(melFiltered)
    melFiltered = melFiltered[1:n_dct,:]
    
    #plt.pcolor(melFiltered)
    ####Normalization
    return melFiltered, mfcc_fs

def spectralCentroid(spect):
   ks = np.linspace(1,spect.shape[0],spect.shape[0])
   numerator = spect * ks[:, np.newaxis]
   numerator= np.sum(numerator,1)
   

#def thresholdPeaks(signal,thresh)
 
def compute_mfccs_frames(frames, fs, hop_size,min_freq,
                  max_freq, num_mel_filts, n_dct):

    win_size = frames.shape[0]
    #Length of spectrogram window
    specLen = int(1+(win_size/2))
    #Sample overlap

    #Take the fft of every frame
    frames = np.fft.fft(frames,win_size,0)
    #Cut them down to size
    frames = np.abs(frames[0:specLen,:])
    
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
        #normalize filters here
        melFiltBank[i,:] = melFiltBank[i,:]/np.sum(melFiltBank[i,:])
        
    melFiltered = np.matmul(melFiltBank,frames)
    melFiltered = 20*np.log10(melFiltered,where=True)
    melFiltered = dct(melFiltered,2,40,0)
    #plt.pcolor(melFiltered)
    melFiltered = melFiltered[1:n_dct,:]
    mfcc_fs = hop_size/fs
    #plt.pcolor(melFiltered)
    ####Normalization
    melFiltered = melFiltered[1:n_dct,:]
    melFiltered = melFiltered - np.min(melFiltered,axis=0)
    melFiltered = melFiltered/np.sum(melFiltered,axis=0) 
    melFiltered = np.nan_to_num(melFiltered)
    mfcc_fs = hop_size/fs
    
    return melFiltered, mfcc_fs

      