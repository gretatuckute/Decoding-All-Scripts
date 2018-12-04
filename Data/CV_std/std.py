# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:28:42 2018

@author: Greta
"""

import pickle 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
import os
import pandas as pd
from scipy.spatial import distance
import scipy.io as sio
# Shape of old raw data is 10350, 550, 32. When std: 10350, 352

os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')

# LOAD EEG DATA #
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']
X = np.reshape(X,[10350,32,60])

y = ASR['Animate']

def std_windows(eeg, time_window=60):
    temp_eeg = []
    for ii in range(eeg.shape[0]):
        temp_epoch = eeg[ii, :, :]
        temp_epoch = np.transpose(temp_epoch)
        temp_eeg.append(
            np.std(temp_epoch.reshape(int(temp_epoch.shape[0] / time_window), time_window, temp_epoch.shape[1]),
                   1).flatten())
    std_eeg = np.array(temp_eeg)
    return std_eeg


#stdtest = std_windows(X,time_window=10)
# 10 ms 1 sample, 6 * 32 = 192 
# If time_window = 10, then one window equals 100 ms (dvs. EEG i seks std bidder) 


def std_overlap_windows(eeg, time_window=10, overlap=0.5):
    temp_eeg = []
    for ii in range(eeg.shape[0]):
        temp_epoch = eeg[ii, :, :]
        temp_epoch = np.transpose(temp_epoch)
        std_channels = []
        for jj in range(temp_epoch.shape[1]):
            channel = temp_epoch[:, jj]

            std_channel = []
            for kk in range(int(time_window * overlap)):
                std_channel.append(channel[kk:(kk + time_window)])

            std_channels.append(np.std(std_channel, 1))

        temp_eeg.append(std_channels)

    temp_eeg_array = np.array(temp_eeg)
    std_temp_eeg = temp_eeg_array.reshape(temp_eeg_array.shape[0], temp_eeg_array.shape[2] * temp_eeg_array.shape[1])
    return std_temp_eeg

#stdoverlap = std_overlap_windows(X,time_window=10, overlap=0.5)
