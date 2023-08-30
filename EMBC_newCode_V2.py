#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:01:54 2021

@author: shoale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
import scipy
import pyaudio
import time
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.io import savemat
from pydub import AudioSegment
from pydub.playback import play
import wave
import librosa
import math
import soundfile as sf
import pyaudio
import wave
import sounddevice as sd
p = pyaudio.PyAudio()

#%% Recording signal
# chunk = 1024  # Record in chunks of 1024 samples
# sample_format = pyaudio.paInt16  # 16 bits per sample
# channels = 2
# fs = 44100  # Record at 44100 samples per second
# seconds = 2
# filename = "output1.wav"
filename1 = '/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/men/m01aw.wav'


sound, fs = sf.read(filename1)
# sound = np.delete(sound, 1, 1)
sound = sound[abs(sound) > 0.002]

song = AudioSegment.from_wav(filename1)
play(song)


#%% Recording signal (2)
# chunk = 1024  # Record in chunks of 1024 samples
# sample_format = pyaudio.paInt16  # 16 bits per sample
# channels = 2
# fs = 44100  # Record at 44100 samples per second
# seconds = 2
# # filename = 'output2.wav'
filename2 = '/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/men/m01oo.wav'
p = pyaudio.PyAudio()  # Create an interface to PortAudio

song = AudioSegment.from_wav(filename2)
play(song)


sound, fs = sf.read(filename2)
# sound = np.delete(sound, 1, 1)
# sound = sound[0:10900]
sound = sound[abs(sound) > 0.002]


#%% Analyzing recording (formants) (1)
# filename = '/Users/shoale/Downloads/SRAL downloads/output1.wav'
# filename = '/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/men/m01ah.wav'
fs = 44100

# Reading, plotting, playing wav file
song = AudioSegment.from_wav(filename1)
# play(song)
sound, fs = sf.read(filename1)
# sound = np.delete(sound, 1, 1)

# Create empty time list
time1 = []
# Add elements to time vector
for j in range(0,len(sound)):
    time1.append(j)
# Get duration of the sound
openSound = wave.open(filename1)
frames = openSound.getnframes()
rate = openSound.getframerate()
duration = frames/float(rate) 
# Create time vector based on seconds
newTime = np.linspace(0,duration,len(time1))


sr = 44100
# Plotting fft of sound
y, sr = librosa.core.load(filename1, sr=sr)
time2 = []
for i in range(0,len(y)):
    time2.append(i)
newTime2 = np.linspace(0,duration,len(time2))
n = len(sound)
T = 1.0/fs
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
fft_out = scipy.fft.fft(y)

#Find intensity of sound
soundRect = abs(y)
b,a = sg.butter(5,20/(fs/2))
soundRectFilt = sg.lfilter(b,a,soundRect)
soundRectFilt = sg.filtfilt(b,a,soundRect,padtype=None,padlen=None)

#Isolate true vowel
isolatedVowel = soundRectFilt[soundRectFilt > 0.0]
isoIndices = np.array([i for i, e in enumerate(soundRectFilt) if e > 0.0])
tIso = newTime[isoIndices[1]:isoIndices[-1]]
# soundRectFiltIso = soundRectFilt[isoIndices[1]:isoIndices[-1]]  #resizing soundrectfilt
soundRectFiltIso = soundRectFilt[isoIndices[1]:isoIndices[len(sound)]]

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


#FFT of isolated vowel
n = len(x1)
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
fft_out1 = scipy.fft.fft(x1)
plt.plot(xf,(2.0/n)*np.abs(fft_out1[0:int(n/2)]))
plt.title('FFT of First Vowel')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(0,5000)
plt.show()

fftFirstVowel = (2.0/n)*np.abs(fft_out1[0:int(n/2)])
# savemat('xf1.mat',{'xf':xf})
# savemat('fftFirstVowel.mat',{'fftFirstVowel':fftFirstVowel})

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
    # errorVectorScaleFactor = 1.5
elif abs(fc1-300)<150 and abs(fc2-870)<150:
    print('Instructor said the vowel "/oo/"!')
    # errorVectorScaleFactor = 1.5
elif abs(fc1-440)<150 and abs(fc2-1020)<150:
    print('Instructor said the vowel "/u/"!')
    # errorVectorScaleFactor = 1.5
elif abs(fc1-730)<100 and abs(fc2-1090)<150:
    print('Instructor said the vowel "/a/"!')
    # errorVectorScaleFactor = 1.5
elif abs(fc1-520)<110 and abs(fc2-1190)<150:
    print('Instructor said the vowel "/uh/"!')
    # errorVectorScaleFactor = 1.5
elif abs(fc1-490)<150 and abs(fc2-1350)<150:
    print('Instructor said the vowel "/er/"!')
    # errorVectorScaleFactor = 1
elif abs(fc1-660)<150 and abs(fc2-1720)<200:
    print('Instructor said the vowel "/ae/"!')
    # errorVectorScaleFactor = 1.5
elif abs(fc1-530)<80 and abs(fc2-1840)<200:
    print('Instructor said the vowel "/e/"!')
    # errorVectorScaleFactor = 0.5
elif abs(fc1-390)<80 and abs(fc2-1990)<200:
    print('Instructor said the vowel "/i/"!')
    # errorVectorScaleFactor = 0.5
elif abs(fc1-270)<150 and abs(fc2-2290)<200:
    print('Instructor said the vowel "/ee/"!')
    # errorVectorScaleFactor = 0.3
else:
    print('Instructor vowel not detected :(')

#%% Analyzing recording (formants) (2)
# filename = '/Users/shoale/Downloads/SRAL downloads/output2.wav'
# filename = '/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/men/m01eh.wav'
# fs = 44100

# Reading, plotting, playing wav file
song = AudioSegment.from_wav(filename2)
# play(song)
sound, fs = sf.read(filename2)
# sound = np.delete(sound, 1, 1)
# Create empty time list
time1 = []
# Add elements to time vector
for j in range(0,len(sound)):
    time1.append(j)
# Get duration of the sound
openSound = wave.open(filename2)
frames = openSound.getnframes()
rate = openSound.getframerate()
duration = frames/float(rate) 
# Create time vector based on seconds
newTime = np.linspace(0,duration,len(time1))


# Plotting fft of sound
y, sr = librosa.core.load(filename2, sr=sr)
time2 = []
for i in range(0,len(y)):
    time2.append(i)
newTime2 = np.linspace(0,duration,len(time2))
n = len(sound)
T = 1.0/fs
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
fft_out = scipy.fft.fft(y)


#Find intensity of sound
soundRect = abs(y)
b,a = sg.butter(5,20/(fs/2))
soundRectFilt = sg.lfilter(b,a,soundRect)
soundRectFilt = sg.filtfilt(b,a,soundRect,padtype=None,padlen=None)


#Isolate true vowel
isolatedVowel = soundRectFilt[soundRectFilt > 0.0]
isoIndices = np.array([i for i, e in enumerate(soundRectFilt) if e > 0.0])
tIso = newTime[isoIndices[1]:isoIndices[-1]]
soundRectFiltIso = soundRectFilt[isoIndices[1]:isoIndices[-1]]
soundRectFiltIso = soundRectFilt[isoIndices[1]:isoIndices[len(sound)]]


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


#FFT of isolated vowel
n = len(x1)
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
fft_out2 = scipy.fft.fft(x1)
plt.plot(xf,(2.0/n)*np.abs(fft_out2[0:int(n/2)]))
plt.title('FFT of Second Vowel')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(0,5000)
plt.show()

fftSecondVowel = (2.0/n)*np.abs(fft_out2[0:int(n/2)])
# savemat('xf2.mat',{'xf':xf})
# savemat('fftSecondVowel.mat',{'fftSecondVowel':fftSecondVowel})


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
    print('Student said the vowel "/ow/"!')
elif abs(fc1-300)<150 and abs(fc2-870)<150:
    print('Student said the vowel "/oo/"!')
elif abs(fc1-440)<150 and abs(fc2-1020)<150:
    print('Student said the vowel "/u/"!')
elif abs(fc1-730)<100 and abs(fc2-1090)<150:
    print('Student said the vowel "/a/"!')
elif abs(fc1-520)<110 and abs(fc2-1190)<150:
    print('Student said the vowel "/uh/"!')
elif abs(fc1-490)<150 and abs(fc2-1350)<150:
    print('Student said the vowel "/er/"!')
elif abs(fc1-660)<150 and abs(fc2-1720)<200:
    print('Student said the vowel "/ae/"!')
elif abs(fc1-530)<80 and abs(fc2-1840)<200:
    print('Student said the vowel "/e/"!')
elif abs(fc1-390)<80 and abs(fc2-1990)<200:
    print('Student said the vowel "/i/"!')
elif abs(fc1-270)<150 and abs(fc2-2290)<200:
    print('Student said the vowel "/ee/"!')
else:
    print('Student vowel undetected :(')


#%% Subtracting instructor from student
# filename = 'output3.wav'
len_fft1, len_fft2 = len(fft_out1), len(fft_out2)
if len_fft1 > len_fft2:
    fft_out1 = fft_out1[1:len_fft2+1]
else:
    fft_out2 = fft_out2[1:len_fft1+1]

fft_difference = (fft_out2 - fft_out1)
n = len(x1)
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
plt.plot(xf,(2.0/n)*np.abs(fft_difference[0:int(n/2)]))
plt.title('FFT of Difference of Vowels')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(0,5000)
plt.ylim(0,0.015)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

fftDifference = (2.0/n)*np.abs(fft_difference[0:int(n/2)])
# savemat('xf3.mat',{'xf':xf})
# savemat('fftDifference.mat',{'fftDifference':fftDifference})


# Converting back to time domain so that formants can be extracted
invFFT = np.fft.ifft(fft_difference)
plt.plot(xf, invFFT[0:len(xf)]*100)
plt.title('Sound of Difference of Recorded Vowels')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

invFFT = invFFT * 100

# samples = invFFT.astype(np.float32).tobytes()
# stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=int(16000),
#                 output=True)
# stream.write(samples)

#%% Analyzing recording (formants) (3)

#Find intensity of sound
soundRect = abs(invFFT)
b,a = sg.butter(5,20/(fs/2))
soundRectFilt = sg.lfilter(b,a,soundRect)
soundRectFilt = sg.filtfilt(b,a,soundRect,padtype=None,padlen=None)


#Isolate true vowel
isolatedVowel = soundRectFilt[soundRectFilt > 0.0]
isoIndices = np.array([i for i, e in enumerate(soundRectFilt) if e > 0.0])
tIso = newTime[isoIndices[1]:isoIndices[-1]]
soundRectFiltIso = soundRectFilt[isoIndices[1]:isoIndices[-1]]


#Second round of isolating vowel
dt = 1/fs
Io = round(tIso[1]/dt)
Iend = round(tIso[-1]/dt)
Io = int(Io)
Iend = int(Iend)
x = sound[Io-1:Iend]
x = np.reshape(x,len(x))
x1 = x

# #Apply hamming window
# x1 = sg.hamming(len(x)) 
# soundRectFiltIso = np.reshape(soundRectFiltIso,len(x1))
# x1 = x * soundRectFiltIso
# x1 = sg.lfilter([1., 0.63], 1, x1)


# #FFT of isolated vowel
# n = len(x1)
# xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
# fft_out3 = scipy.fft.fft(x1)


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

# Extracting formants
fc1 = formants[0]
fc2 = formants[1]
fc3 = formants[2]

#%% Adding formants from difference vector to student vector
# fft_difference_1 = fft_difference[abs(fft_difference)]
count = 1
boolX = []
xf = np.linspace(0.0,1.0/(2.0*T),n)
maxDifference = 100
for i in xf:
    if abs(fc1-i) < maxDifference:
        boolX.append(1)
    elif abs(fc2-i) < maxDifference:
        boolX.append(1)
    elif abs(fc3-i) < maxDifference:
        boolX.append(1)
    else:
        boolX.append(0)
    count  = count + 1
    
count = 1
fft_forAdding = []
for j in boolX:
    if j == 1:
        fft_forAdding.append(fft_difference[count])
    elif j == 0:
        fft_forAdding.append(0)
    count = count + 1

fft_forAdding.append(0)
fft_forAdding.append(0)
fft_forAdding = np.array(fft_forAdding)
fft_forAdding = abs(fft_forAdding)

# plt.plot(xf[1:len(fft_forAdding)],fft_forAdding[1:len(fft_forAdding)-2]) # making arrays the same size so that they can be plotted
# plt.show() #two data points at the end can be seen as obsolete


# %% Adding error formants to student recording
errorVectorScaleFactor = 8
fft_forAdding = [i * errorVectorScaleFactor for i in fft_forAdding] # scaling error vector
fft_out2 = np.insert(fft_out2,0,0)
fft_out2 = np.insert(fft_out2,0,0)
fft_studentPlusError = fft_out2 + fft_forAdding

# Calculating euclidian distance between set scalar and actual scalar
errorVectorScaleFactor2 = 1
fft_forAdding2 = [i * errorVectorScaleFactor for i in fft_forAdding]
fft_studentPlusError2 = fft_out2 + fft_forAdding2

# savemat('fft_forAdding.mat',{'fft_forAdding':fft_forAdding})

eucDist = np.linalg.norm(fft_studentPlusError - fft_studentPlusError2)
print('Euclidian Distance between set and actual error augmentations: ', eucDist)

# Plotting FFT of student plus error array
plt.plot(xf, abs(fft_studentPlusError[1:len(fft_studentPlusError)-1]))
plt.title('FFT of Student Plus Error')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# plt.ylim(0,5)
plt.xlim(0,5000)
plt.show()

fftStudentPlusError = abs(fft_studentPlusError[1:len(fft_studentPlusError)-1])
# savemat('xf4.mat',{'xf':xf})
# savemat('fftStudentPlusError.mat',{'fftStudentPlusError':fftStudentPlusError})


#%% Convert back to time domain so we can play it
invFFT2 = np.fft.ifft(fft_studentPlusError)
plt.plot(xf, invFFT2[0:len(xf)])
plt.title('Error Augmented Sound')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

soundOfDiffs = invFFT2[0:len(xf)]
# savemat('xf5.mat',{'xf':xf})
# savemat('soundOfDiffs.mat',{'soundOfDiffs':soundOfDiffs})


invFFT2 = invFFT2[abs(invFFT2) > 0.005]
invFFT2 = invFFT2 * 10

#%% Playing sound with error
samples = invFFT2.astype(np.float32).tobytes()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=int(11000),
                output=True)
stream.write(samples)

# write('errorAug.wav', 10, invFFT2.astype(np.float32))
# n, sr2 = librosa.load('errorAug.wav', sr=1)









