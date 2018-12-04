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

os.chdir('C:/Users/Greta/Documents/GitHub/Project-MindReading/data/ASR/')

# LOAD EEG DATA #
ASR = sio.loadmat('ASRfile')
X = ASR['A']

# LOAD y #
y = np.load('y.npy')

##### Testing on model with first subject out as test ######
# First iteration of my SVM crossvalidation with leave one subject out
cv = list(range(0,len(y),690))
test = list(range(ii, ii+690))
train = np.delete(list(range(0, len(y))), test, 0)

X=X[train]

######### Matrices and gamma SVM evaluation ##############
M = distance.pdist(X,'cosine')
#Msquare = distance.squareform(M)
Mexp = np.exp(-M**2)
Mexpsquare = distance.squareform(Mexp)

Mexp_gamma = np.exp((np.square(1/400))*(-(np.square(M))))
Mexpsquare_gamma = distance.squareform(Mexp_gamma) # THIS SHOULD BE THE NxN TRAINING KERNEL MATRIX

with open("dualcoef_test.pckl", 'rb') as pickleFile:
    dualcoef = pickle.load(pickleFile)
    
with open("SVM_ASR_test.pckl", 'rb') as pickleFile2:
    model = pickle.load(pickleFile2)
    

###### Transpose X to achieve training examples in columns ######
Xt = np.transpose(X)

# COMPUTE SENSITIVITY
# map=X*diag(alpha)*K−X*diag(alpha*K);
# s=sum(map.*map,2)/numel(alpha);

X1 = Xt
K = Mexpsquare_gamma
alpha = dualcoef

map1 = np.matmul(X1,np.matmul(np.diag(alpha3),K))-(np.matmul(X1,(np.diag(np.matmul(alpha3,K)))))
s = np.sum(np.square(map1),axis=1)/np.size(alpha)

plt.plot(s)

# 1600 is the resampled EEG signal. 
