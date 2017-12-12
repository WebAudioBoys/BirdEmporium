# -*- coding: utf-8 -*-
#Spyder Editor

#This is a temporary script file.
#"""
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.signal import medfilt

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

def freq2mel(freq):
    melval = 1127.01028 * np.log(1 + (freq/700));
    return melval

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

    


def createThreshold(array,filtLen):
    threshold = medfilt(array, filtLen)
    return threshold

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
        

def mel2freq(melval):
    hzval = 700 * (np.expm1(melval/1127.01028));
    return hzval

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
   
def findPeaks(signal):
    signal = np.diff(signal) 
    #Returns indices of zero cross in diff
    np.append([0],signal)
    diffZC = np.where((np.diff(np.sign(signal))))[0]
    
    #Only find upper peaks
    diffZC = -1 * diffZC
    diffZC = (diffZC + np.abs(diffZC))/2
    
#    for i in range(0,len(diffZC)):
#        if signal[diffZC[i]] > thresh:
#            np.append(output,signal[diffZC[i]])
#        else:
#            
    
    return diffZC

#def thresholdPeaks(signal,thresh)
 
def compute_mfccs_frames(frames, fs, hop_size,min_freq,
                  max_freq, num_mel_filts, n_dct):

    win_size = frames.shape[0]
    #Length of spectrogram window
    specLen = int(1+(frames.shape[0]/2))
    #Sample overlap

    #Take the fft of every frame
    frames = np.fft.fft(frames,win_size,0)
    #Cut them down to size
    frames = np.abs(frames[0:specLen,:])
    
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
    melFiltered = 20*np.log10(melFiltered)
    melFiltered = dct(melFiltered,2,40,0)
    #plt.pcolor(melFiltered)
    melFiltered = melFiltered[1:n_dct,:]
    mfcc_fs = hop_size/fs
    #plt.pcolor(melFiltered)
    ####Normalization
    return melFiltered, mfcc_fs      
       

    
    

    
    



 



    
    
    





