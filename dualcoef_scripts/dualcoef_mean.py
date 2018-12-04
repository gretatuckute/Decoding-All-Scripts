# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:27:13 2018

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

os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')

# LOAD EEG
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)


######### Matrices and gamma SVM evaluation ##############
M = distance.pdist(X,'euclidean')
#Msquare = distance.squareform(M)
Mexp = np.exp(-M**2)
Mexpsquare = distance.squareform(Mexp)

Mexp_gamma = np.exp(0.00005*(-(np.square(M)))) # Is this correct? 0.00025 should not be squared?
Mexpsquare_gamma = distance.squareform(Mexp_gamma) # THIS SHOULD BE THE NxN TRAINING KERNEL MATRIX

os.chdir('C:/Users/Greta/Documents/GitHub/decoding/pckl/')
with open("dualcoef_mean_parameter.pckl", 'rb') as pickleFile:
    dualcoef_mean = pickle.load(pickleFile)

    
with open("SVM_model_mean_parameter.pckl", 'rb') as pickleFile2:
    SVM_model_mean = pickle.load(pickleFile2)
    
support_array = SVM_model_mean.support_




# I have to end up with an array size 10350, thus have to inset zero's 

# If idx of dualcoef does not equal the number in support_array, eg 0 != 31:
# append 0 to new list 
# if idx of dualcoef equals number in support_array, e.g. 31 == 31
# append value in dualcoef to new list 
            
# if the count of the idx..
            
dualcoef_mean=np.squeeze(dualcoef_mean)


seq=np.zeros(10350)
seq[support_array]=dualcoef_mean

# Number of zero's in the dual coef array:
10350-9146

import collections

collections.Counter(seq) #1204 zero values


###### Transpose X to achieve training examples in columns ######
Xt = np.transpose(X)

# COMPUTE SENSITIVITY
# map=X*diag(alpha)*K−X*diag(alpha*K);
# s=sum(map.*map,2)/numel(alpha);

X1 = Xt
K = Mexpsquare_gamma
alpha = seq #Python can't use the alpha shape (1, 9660)
alpha1 = np.squeeze(alpha)

map1 = np.matmul(X1,np.matmul(np.diag(alpha1),K))-(np.matmul(X1,(np.diag(np.matmul(alpha1,K)))))
s = np.sum(np.square(map1),axis=1)/np.size(alpha) #Px1 vector

s_res = np.reshape(s,[32,60])
np.save('s_map_mean_NEW.npy',s_res)
sio.savemat('s_map_mean.mat', mdict={'map': s_res})



channel_vector = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4',
                  'Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5',
                  'FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']

time_vector = ['-100','0','100','200','300','400','500']

plt.matshow(s_res)
plt.xlabel('Time (ms)')
plt.xticks(np.arange(0,60,10),time_vector)
plt.yticks(np.arange(32),channel_vector)
plt.ylabel('Electrode number ')
plt.colorbar()
#plt.yticks(np.arange(len(gamma_range)), gamma_range)
#plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
plt.title('Sensitivity map SVM RBF kernel - ASR 100Hz - mean CV parameters')
plt.show()

########### CHECK WHETHER DUAL COEF CHANGES WHEN y={1, -1} vs y={1,0}
with open("SVM_model_y1-1_mean_parameter_subj2.pckl", 'rb') as pickleFile:
    SVM_model_median = pickle.load(pickleFile)


with open("SVM_model_mean_parameter_subj2.pckl", 'rb') as pickleFile2:
    SVM_model_median_y = pickle.load(pickleFile2)

dual = SVM_model_median.dual_coef_
dual_y = SVM_model_median_y.dual_coef_

np.array_equal(dual, dual_y) # The arrays are equal
