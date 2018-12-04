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
import pandas as pd
import datetime

# Load data
# os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

# Change y to 1 and -1
#y[y < 1] = -1
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)

# Parameters to iterate through
# C_2d_range = [0.05, 0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 5, 10]
C_2d_range = [0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 5, 10, 15]

#gamma_2d_range = [0.00005, 0.00025, 0.0005, 0.00075, 0.001]
gamma_2d_range = [1/2000000, 1/400000, 1/200000, 1/40000, 1/20000, 1/4000, 1/2000, 1/400, 1/200, 1/40]
random_state = np.random.RandomState(0)

# subj = 0 denotes no. 1, and thus the first two ones out

sub1 = 0
sub2 = 14

df = pd.DataFrame(columns=['test1_subj_no', 'test2_subj_no', 'C value', 'Gamma value', 'scores_train', 'scores_test1', 'scores_test2'])

cv = list(range(0,len(y),690))
ii = cv[sub1] # ii denotes which subject is used for validation 
jj = cv[sub2] # jj denotes test subject

sub_out1 = list(range(ii, ii+690))
sub_out2 = list(range(jj, jj+690))

subs_out = sub_out1 + sub_out2

train = np.delete(list(range(0, len(y))), subs_out, 0)

X_train=X[train]
y_train=y[train]

X_out1=X[sub_out1]
y_out1=y[sub_out1]

X_out2=X[sub_out2]
y_out2=y[sub_out2]

count = 0

print('============ Data Loaded ============')
print('X train shape: ' + str(X_train.shape))
print('y train shape: ' + str(y_train.shape))
print('Subject out1: ' + str(sub1+1))
print('Subject out2: ' + str(sub2+1))

for gamma in gamma_2d_range:
    for C in C_2d_range:
        classifier = SVC(C=C, gamma=gamma, random_state=random_state)
        clf = classifier.fit(X_train, y_train)
    
        scores_train = clf.score(X_train, y_train)
        
        scores_test1 = clf.score(X_out1, y_out1)
        scores_test2 = clf.score(X_out2, y_out2)
        
        df.loc[count]=[sub1+1, sub2+1, C, gamma, scores_train, scores_test1, scores_test2]
    
        count += 1
        print(count)

            
df.to_excel('CV_val+test_' + str(sub1+1) + '_' + str(sub2+1) + '.xlsx')

    

