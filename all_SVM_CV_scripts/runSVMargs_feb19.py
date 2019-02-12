# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:23:18 2018

@author: Greta, grtu@dtu.dk

#EXAMPLE FUNCTION CALL
#	python runSVM.py -s 0

"""

#Imports
import argparse
import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pickle 
import pandas as pd
import datetime
import os
from scipy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


date = str(datetime.datetime.now())
date_str = date.replace(' ','-')
date_str = date_str.replace(':','.')
date_str = date_str[:-10]

#Constructing the parser and parse the arguments
parser = argparse.ArgumentParser(description='Takes subject number for test set (-s)')
parser.add_argument('-s','--subject', required=True, default=None,help='Specify which subject to leave out as test set. 0 = subject 1')
args = vars(parser.parse_args()) 

subj = args['subject']
subj = int(subj)

# Load data
#os.chdir('C:/Users/Greta/Documents/GitHub/Decoding-All-Scripts/Data/ASR/')
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

X1 = np.reshape(X,[10350,32,60])

y = ASR['Animate']
y = np.squeeze(y)

# Change y to 1 and -1
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)


#%%
def channelPCA(eegdata, no_PCs):
    '''
    Computes channel-wise PCA (using an SVD mapping) of a 3D EEG data matrix ([trials,channels,samples]).
    
    # Input
    - eegdata: Epoched EEG data in the following format: ([trials, time samples, channels]).
    - no_PCs: number of principal components (PC) to use for the PCA projection of the EEG data.
    
    # Output
    - PCA_eeg: PCA projection of the EEG ([trials,channels*number of PCs])
    - V_eeg: V matrices from the SVD (one for each channel SVD)
    - means: Channel-wise mean values, used for standardizing each channel.

    '''
    
    no_trials = eegdata.shape[0]
    no_chs = eegdata.shape[1]
    no_samples = eegdata.shape[2]
    
    PCA_eeg = np.zeros((no_trials,no_chs,no_PCs))
    V_eeg = np.zeros((no_chs,no_samples,no_samples))
    means = []
    
    for ch in range(no_chs):
        channel = eegdata[:,ch,:]
        channel_stand = np.copy(channel)
        channel_mean = np.mean(channel, axis=0)# To standardize sample-wise
        means.append(channel_mean)
        
        channel_stand = channel_stand - channel_mean
        
        U,S,V = svd(channel_stand,full_matrices=False)
        V_eeg[ch,:,:] = V
            
        Z = np.dot(channel_stand, V.T)
        # print(Z.shape)
        Z = Z[:,0:no_PCs]
        PCA_eeg[:,ch,:] = Z
    
    PCA_eeg = np.reshape(PCA_eeg,[no_trials,no_chs*no_PCs])
    
    return PCA_eeg, V_eeg, means

def findPCs(epochs,cat,subj):
    '''
    
    CURRENT VERSION DOES NOT LOOP OVER SUBJECTS, but have to input
    
    Cross-validates a number of principal components (PCs) to find the optimal number of PCs based on a simple logistic regression classifier for a binary classification task.
    Computes channel-wise PCA (using an SVD mapping) using channelPCA function.
    OBS: 1) Possible to validate using LDA, 2) Possible to z-score training and test sets.
    
    # Input
    - epochs: Epoched EEG data in the following format: ([trials, time samples, channels]).
    - cat: Binary category list.
    
    # Output
    - max_PC: Optimum number of PCs based on validation sets and logistic regression classification.
    - score_train: Training score with the optimum number of PCs.
    - score_test: Test score with the optimum number of PCs.
    
    '''
        
    PC_span = [4,6,8,10,12,14,16,18,20,26,40]
    len_PCs = len(PC_span)
    
    C_span = [0.05, 0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 5, 10]
    len_C = len(C_span)

    
    cv = list(range(0,len(cat),690))
    ii = cv[subj]
    
    test = list(range(ii, ii+690))
    # train = np.delete(list(range(0, len(y))), test + val, 0)
    train = np.delete(list(range(0, len(cat))), test, 0)

    X_train = epochs[train]
    X_test = epochs[test]
    
    y_full = np.asarray(cat)
    y = y_full[train]
    y_test = y_full[test]
    
    val_acc = np.zeros((len_PCs,14))
    
    for count, value in enumerate(PC_span): # Number of PCs loop
        
        # Compute PCA 
        X_train_pc, V_train, means_train = channelPCA(X_train,no_PCs=value)
        
        # X_train_pc = stats.zscore(X_train_pc, axis=1)
        
        c = 0
        
        for jj in range(14): # CV validation loop
            
            indVal = list(range(jj,jj+690))
            indTrain = np.delete(list(range(0, len(train))), indVal, 0)

            X_val = X_train_pc[indVal,:]
            y_val = y[indVal]
            
            indTrain = np.asarray(range(len(train)))
            indTrain = np.delete(indTrain,indVal,axis=0)
            
            X_train2 = X_train_pc[indTrain,:]
            y_train = y[indTrain]
        
            classifier = SVC(random_state=0)
            
            clf = classifier.fit(X_train2, y_train)
            y_val_pred = clf.predict(X_val)
                
            val_acc[count,c] = metrics.accuracy_score(y_val, y_val_pred)
            
            c += 1
            
    mean_val_accs = np.mean(val_acc,axis=1)       
    max_acc_idx = np.argmax(mean_val_accs)
    
    max_PC = PC_span[max_acc_idx]
    print('Optimal number of PCs is: ' + str(max_PC))
    
    # Test on the test set
    classifier_test = SVC(random_state=0)
    # classifier_test = LinearDiscriminantAnalysis()
    
    X_train_optimum, V_optimum, means_optimum = channelPCA(X_train,no_PCs=max_PC) 
    
    clf_test = classifier_test.fit(X_train_optimum,y)

    score_train = clf_test.score(X_train_optimum,y)
    
    PCA_eeg = np.zeros((690,32,max_PC))

    # Compute PCA on the test set based on the projection from the training set 
    for ch in range(32):
        channel = X_test[:,ch,:]
        channel_stand = np.copy(channel)
        channel_stand = channel_stand - means_optimum[ch]
        #U,S,V = svd(channel,full_matrices=False)
        Z = np.dot(channel_stand, V_optimum[ch].T)
        Z = Z[:,0:max_PC]
        PCA_eeg[:,ch,:] = Z
    
    X_test_optimum = np.reshape(PCA_eeg,[690,32*max_PC])
    
    # X_test_optimum = stats.zscore(X_test_optimum, axis=1)
    
    score_test = clf_test.score(X_test_optimum,y_test)
    
    print('Test accuracy: ' + str(score_test))
    
    return max_PC, score_train, score_test

    
#%%
    
def findPCs2(epochs,cat,subj):
    '''
    
    CURRENT VERSION DOES NOT LOOP OVER SUBJECTS, but have to input
    
    Cross-validates a number of principal components (PCs) to find the optimal number of PCs based on a simple logistic regression classifier for a binary classification task.
    Computes channel-wise PCA (using an SVD mapping) using channelPCA function.
    OBS: 1) Possible to validate using LDA, 2) Possible to z-score training and test sets.
    
    # Input
    - epochs: Epoched EEG data in the following format: ([trials, time samples, channels]).
    - cat: Binary category list.
    
    # Output
    - max_PC: Optimum number of PCs based on validation sets and logistic regression classification.
    - score_train: Training score with the optimum number of PCs.
    - score_test: Test score with the optimum number of PCs.
    
    '''
        
    PC_span = [4,6,10,12,16,20,26,40]
    len_PCs = len(PC_span)
    
    C_span = [0.5, 1, 2, 5, 10]
    len_C = len(C_span)

    
    cv = list(range(0,len(cat),690))
    ii = cv[subj]
    
    test = list(range(ii, ii+690))
    # train = np.delete(list(range(0, len(y))), test + val, 0)
    train = np.delete(list(range(0, len(cat))), test, 0)
    
    print('Length of training set: ' + str(len(train)))

    X_train = epochs[train]
    X_test = epochs[test]
    
    y_full = np.asarray(cat)
    y = y_full[train]
    y_test = y_full[test]
    
    val_acc = np.zeros((len_PCs,14,len_C))
    
    for count, value in enumerate(PC_span): # Number of PCs loop
        
        # Compute PCA 
        X_train_pc, V_train, means_train = channelPCA(X_train,no_PCs=value)
        
        # X_train_pc = stats.zscore(X_train_pc, axis=1)
        
        c = 0
        
        print('PC iteration number: ' + str(c))
        
        for jj in range(14): # CV validation loop
            
            indVal = list(range(jj,jj+690))
            indTrain = np.delete(list(range(0, len(train))), indVal, 0)

            X_val = X_train_pc[indVal,:]
            y_val = y[indVal]
            
            indTrain = np.asarray(range(len(train)))
            indTrain = np.delete(indTrain,indVal,axis=0)
            
            X_train2 = X_train_pc[indTrain,:]
            y_train = y[indTrain]
            
            c_count = 0
            
            for Cc in C_span:
                
                print('C iteration number: ' + str(c_count))
                
                classifier = SVC(gamma=5e-5,C=Cc,random_state=0)
                
                clf = classifier.fit(X_train2, y_train)
                y_val_pred = clf.predict(X_val)
                    
                val_acc[count,c,c_count] = metrics.accuracy_score(y_val, y_val_pred)
                
                c_count += 1
        
            val_acc[count,c] = metrics.accuracy_score(y_val, y_val_pred)
            
            c += 1
    
    mean_val_accs = np.mean(val_acc,axis=1)
    
    # Find optimum number of PCs
    max_PC_vals = np.mean(mean_val_accs,axis=1)
    max_PC_idx = np.argmax(max_PC_vals)
    
    # Find optimum value of c
    max_c_vals = np.mean(mean_val_accs,axis=0)
    max_c_idx = np.argmax(max_c_vals)
    
    max_PC = PC_span[max_PC_idx]
    max_c = C_span[max_c_idx]
    print('Optimal number of PCs is: ' + str(max_PC) + ' with an optimum c value of: ' + str(max_c))
    
    # Test on the test set
    classifier_test = SVC(gamma=5e-5,C=max_c,random_state=0)
    # classifier_test = LinearDiscriminantAnalysis()
    
    X_train_optimum, V_optimum, means_optimum = channelPCA(X_train,no_PCs=max_PC) 
    
    clf_test = classifier_test.fit(X_train_optimum,y)

    score_train = clf_test.score(X_train_optimum,y)
    
    PCA_eeg = np.zeros((690,32,max_PC))

    # Compute PCA on the test set based on the projection from the training set 
    for ch in range(32):
        channel = X_test[:,ch,:]
        channel_stand = np.copy(channel)
        channel_stand = channel_stand - means_optimum[ch]
        #U,S,V = svd(channel,full_matrices=False)
        Z = np.dot(channel_stand, V_optimum[ch].T)
        Z = Z[:,0:max_PC]
        PCA_eeg[:,ch,:] = Z
    
    X_test_optimum = np.reshape(PCA_eeg,[690,32*max_PC])
    
    # X_test_optimum = stats.zscore(X_test_optimum, axis=1)
    
    score_test = clf_test.score(X_test_optimum,y_test)
    
    print('Test accuracy: ' + str(score_test))
    
    return val_acc, max_PC, max_c, score_train, score_test

val_acc, max_PC, max_c, score_train, score_test = findPCs2(epochs=X1,cat=y,subj=subj)

grand_lst = [val_acc,max_PC,max_c,score_train,score_test]

fname = 'SVM_PCA_subj_' + str(subj) + '.pkl'
with open(fname, "wb") as fout:
    pickle.dump(grand_lst, fout)
    
    
