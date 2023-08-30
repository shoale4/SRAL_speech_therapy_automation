#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:08:59 2021

@author: shoale
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
import scipy
import pyaudio
from pydub import AudioSegment
import wave
import librosa
import math
import soundfile as sf
p = pyaudio.PyAudio()

def gettingFormants(filename):

    fs = 44100
    
    # Reading, plotting, playing wav file
    song = AudioSegment.from_wav(filename)
    # play(song)
    sound, fs = sf.read(filename)
    # sound = np.delete(sound, 1, 1)
    # Create empty time list
    time1 = []
    # Add elements to time vector
    for j in range(0,len(sound)):
        time1.append(j)
    # Get duration of the sound
    openSound = wave.open(filename)
    frames = openSound.getnframes()
    rate = openSound.getframerate()
    duration = frames/float(rate) 
    # Create time vector based on seconds
    newTime = np.linspace(0,duration,len(time1))
    # Plot new time list with sound data
    plt.plot(newTime, sound)
    plt.title('Original Sound')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
    # Plotting fft of sound
    y, sr = librosa.core.load(filename, sr=44100)
    time2 = []
    for i in range(0,len(y)):
        time2.append(i)
    newTime2 = np.linspace(0,duration,len(time2))
    n = len(sound)
    T = 1.0/fs
    xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
    fft_out = scipy.fft.fft(y)
    # plt.figure(2)
    # plt.plot(xf,(2.0/n)*np.abs(fft_out[0:int(n/2)]))
    # plt.title('FFT of Sound')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.show()
    
    #Find intensity of sound
    soundRect = abs(y)
    b,a = sg.butter(5,20/(fs/2))
    soundRectFilt = sg.lfilter(b,a,soundRect)
    soundRectFilt = sg.filtfilt(b,a,soundRect,padtype=None,padlen=None)
    # plt.plot(newTime,soundRectFilt)
    # plt.title('Rectification of Sound')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.show()
    
    #Isolate true vowel
    isolatedVowel = soundRectFilt[soundRectFilt > 0.0]
    isoIndices = np.array([i for i, e in enumerate(soundRectFilt) if e > 0.0])
    tIso = newTime[isoIndices[1]:isoIndices[-1]]
    soundRectFiltIso = soundRectFilt[isoIndices[1]:isoIndices[-1]]
    # plt.plot(tIso,soundRectFiltIso)
    # plt.title('First Round of Vowel Isolation')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.show()
    
    #Second round of isolating vowel
    dt = 1/fs
    Io = round(tIso[1]/dt)
    Iend = round(tIso[-1]/dt)
    Io = int(Io)
    Iend = int(Iend)
    x = sound[Io-1:Iend]
    x = np.reshape(x,len(x))
    
    #Apply hamming window
    x1 = sg.hamming(len(x)) 
    soundRectFiltIso = np.reshape(soundRectFiltIso,len(x1))
    x1 = x * soundRectFiltIso
    x1 = sg.lfilter([1., 0.63], 1, x1)
    # plt.plot(tIso,x1)
    # plt.title('Second Round of Vowel Isolation (Hamming Window)')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.show()
    
    #FFT of isolated vowel
    n = len(x1)
    xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
    fft_out_final = scipy.fft.fft(x1)
    plt.plot(xf,(2.0/n)*np.abs(fft_out_final[0:int(n/2)]))
    plt.title('FFT of First Vowel')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.xlim(0,5000)
    plt.show()
    
    #Find LPC coefficients
    A = librosa.core.lpc(x1,30)
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]
    
    #Phase angle
    angz = np.arctan2(np.imag(rts), np.real(rts))
    
    #Get formants
    formants = sorted(angz * (fs / (2 * math.pi)))
    
    # Discarding any values of 0 (if any)
    formants = list(filter(lambda num: num != 0, formants))
    
    #Printing F1-F3
    print('[F1, F2, F3]:', formants[0:3])
    
    # Detecting the word you said
    fc1 = formants[0]
    fc2 = formants[1]
    fc3 = formants[2]
    
    if abs(fc1-570)<100 and abs(fc2-840)<150:
        print('Instructor said the vowel "/ow/"!')
        errorVectorScaleFactor = 1.5
    elif abs(fc1-300)<150 and abs(fc2-870)<150:
        print('Instructor said the vowel "/oo/"!')
        errorVectorScaleFactor = 1.5
    elif abs(fc1-440)<150 and abs(fc2-1020)<150:
        print('Instructor said the vowel "/u/"!')
        errorVectorScaleFactor = 1.5
    elif abs(fc1-730)<100 and abs(fc2-1090)<150:
        print('Instructor said the vowel "/a/"!')
        errorVectorScaleFactor = 1.5
    elif abs(fc1-520)<110 and abs(fc2-1190)<150:
        print('Instructor said the vowel "/uh/"!')
        errorVectorScaleFactor = 1.5
    elif abs(fc1-490)<150 and abs(fc2-1350)<150:
        print('Instructor said the vowel "/er/"!')
        errorVectorScaleFactor = 1
    elif abs(fc1-660)<150 and abs(fc2-1720)<200:
        print('Instructor said the vowel "/ae/"!')
        errorVectorScaleFactor = 1.5
    elif abs(fc1-530)<80 and abs(fc2-1840)<200:
        print('Instructor said the vowel "/e/"!')
        errorVectorScaleFactor = 0.5
    elif abs(fc1-390)<80 and abs(fc2-1990)<200:
        print('Instructor said the vowel "/i/"!')
        errorVectorScaleFactor = 0.5
    elif abs(fc1-270)<150 and abs(fc2-2290)<200:
        print('Instructor said the vowel "/ee/"!')
        errorVectorScaleFactor = 0.3
    else:
        print('Instructor vowel not detected :(')
        
    
    return fft_out_final, x1, T, newTime