#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:15:57 2021

@author: shoale
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:26:09 2020

@author: shoale
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:14:02 2020

@author: shoale
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
import scipy
import pyaudio
import librosa
import math
import readSignal as readSignal
import gettingFormants as gettingFormants
p = pyaudio.PyAudio()


fs = 41000

#%% Reading and getting formants of signals in signals

sig0 = readSignal.readSignal('output1.wav')
x0 = gettingFormants.gettingFormants('output1.wav')
fft_out1 = x0[0]

sig1 = readSignal.readSignal('output2.wav')
y0 = gettingFormants.gettingFormants('output2.wav')
fft_out2 = y0[0]


#%% Subtracting instructor from student
x1 = x0[1]
T = x0[2]
newTime = x0[3]

fft_difference = (fft_out2 - fft_out1) 
n = len(x1)
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
plt.plot(xf,abs((2.0/n)*np.abs(fft_difference[0:int(n/2)])))
plt.title('FFT of Difference of Vowels')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(0,5000)
plt.ylim(0,0.03)
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Converting back to time domain so that formants can be extracted
invFFT = np.fft.ifft(fft_difference)
plt.plot(xf, invFFT[0:len(xf)]*1000)
plt.title('Sound of Differences of Recorded Vowels')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


#%% Analyzing recording (formants) (3)

#Find intensity of sound
soundRect = abs(invFFT)
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
x = invFFT[Io-1:Iend]
x = np.reshape(x,len(x))

#Apply hamming window
x1 = sg.hamming(len(x)) 
# soundRectFiltIso = np.reshape(soundRectFiltIso,len(x1))
soundRectFiltIso = soundRectFiltIso[0:len(x1)]
x1 = x * soundRectFiltIso
x1 = sg.lfilter([1., 0.63], 1, x1)


#FFT of isolated vowel
n = len(x1)
xf = np.linspace(0.0,1.0/(2.0*T),int(n/2))
fft_out3 = scipy.fft.fft(x1)
plt.plot(xf,(2.0/n)*np.abs(fft_out3[0:int(n/2)]))
# plt.title('FFT of Isolated Vowel')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.xlim(0,5000)
# plt.show()

#Find LPC coefficients
x1 = x1.real
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

#%% Adding formants from difference vector to student vector
# fft_difference_1 = fft_difference[abs(fft_difference)]
count = 1
boolX = []
xf = np.linspace(0.0,1.0/(2.0*T),n)
maxDifference = 200
# for i in xf:
#     if abs(fc1-i) < maxDifference:
#         boolX.append(1)
#     elif abs(fc2-i) < maxDifference:
#         boolX.append(1)
#     elif abs(fc3-i) < maxDifference:
#         boolX.append(1)
#     else:
#         boolX.append(0)
#     count  = count + 1

# centroid1 = (fc1+fc2)/2
# centroid2 = (fc2+fc3)/2
# for i in xf:
#     if abs(centroid1-i) < maxDifference:
#         boolX.append(1)
#     elif abs(centroid2-i) < maxDifference:
#         boolX.append(1)
#     else:
#         boolX.append(0)
#     count = count + 1
    
        
fft_difference = fft_difference * 10
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

# fft_forAdding = abs(np.array(fft_forAdding))

# plt.plot(xf[1:len(fft_forAdding)],fft_forAdding[1:len(fft_forAdding)-2])
# plt.xlim(0,4000)
# plt.ylim(0,500)
# plt.show()


# %% Adding error formants to student recording
errorVectorScaleFactor = 0.2
fft_forAdding_scaled = [i * errorVectorScaleFactor for i in fft_forAdding] # scaling error vector
fft_forAdding_scaled = np.array(fft_forAdding_scaled)
fft_out2 = fft_out2[0:len(fft_forAdding_scaled)]
fft_studentPlusError = fft_forAdding_scaled + fft_out2
fft_studentPlusError = fft_difference
plt.plot(xf, abs(fft_studentPlusError[0:len(xf)]))
plt.title('FFT of Student Plus Error')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim(0,500)
plt.xlim(0,5000)
plt.show()

#%% Convert back to time domain so we can play it
invFFT2 = np.fft.ifft(fft_studentPlusError)*10
plt.plot(xf, invFFT2[0:len(xf)])
plt.title('Sound of Differences of Recorded Vowels')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

invFFT2 = invFFT2[abs(invFFT2) > 0.005]

#%% Playing sound with error
samples = invFFT2.astype(np.float32).tobytes()
stream = p.open(format=pyaudio.paFloat32,
                channels=2,             # stereo
                rate=int(41000/2),
                output=True)
stream.write(samples)









