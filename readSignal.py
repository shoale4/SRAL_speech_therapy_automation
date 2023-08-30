#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:58:59 2021

@author: shoale
"""

import soundfile as sf

def readSignal(filename):
    sound, fs = sf.read(filename)
    # sound = np.delete(sound, 1, 1)
    sound = sound[abs(sound) > 0.002]
    
    return sound
    
# readSignal('output1.wav')