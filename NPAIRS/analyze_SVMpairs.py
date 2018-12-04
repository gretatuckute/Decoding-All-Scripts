# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:26:09 2018

@author: Greta
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\decoding\\pckl\\SVM_pairs\\100_new')

ef = np.load('effect_smap_new_100.npy')
mean = np.load('mean_smap_new_100.npy')
std = np.load('std_smap_new_100.npy')

#sio.savemat('ef25_SVM_GT.mat', mdict={'ef': ef})

# Try to smooth the std error plot, before using it to get the effect size plot 
from scipy import signal
kernel = np.ones([60,1])
kernel2 = np.ones([3,5]) # Horizontal smoothing 
conv = signal.convolve2d(std, kernel2, boundary='symm', mode='same')

plt.matshow(conv)

# finde mean paa effect size. Beregn middelvaerdien, og divider med den.
std_meanval = np.mean(std)

# interpolation i scalp map. 

# Create new ef plot
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\decoding\\pckl') # Load the real sensitivity map for 15 subjs
smap_15 = np.load('s_map_mean_NEW.npy')

effect_smap = np.divide(smap_15,std_meanval)

sio.savemat('effectsize_sensitivitymap.mat', mdict={'ef': effect_smap})



channel_vector = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4',
                  'Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5',
                  'FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']

time_vector = ['-100','0','100','200','300','400','500']

plt.matshow(effect_smap)
plt.xlabel('Time (ms)')
plt.xticks(np.arange(0,60,10),time_vector)
plt.yticks(np.arange(32),channel_vector)
plt.ylabel('Electrode number ')
plt.colorbar()
#plt.yticks(np.arange(len(gamma_range)), gamma_range)
#plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
plt.title('Effect size 100 iterations')
#plt.title('Std error, 100 iterations')
plt.show()

