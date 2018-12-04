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

#Constructing the parser and parse the arguments
# =============================================================================
# parser = argparse.ArgumentParser(description='Takes subject number for test set (-s)')
# parser.add_argument('-s','--subject', required=True, default=None,help='Specify which subject to leave out as test set. 0 = subject 1')
# args = vars(parser.parse_args()) 
# 
# subj = args['subject']
# subj = int(subj)
# =============================================================================

subj = 1 

# Load data
os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')
ASR = sio.loadmat('ASRnew')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

# Parameters to iterate through
C_2d_range = [1.0]
gamma_2d_range = [0.0025]

random_state = np.random.RandomState(0)

df = pd.DataFrame(columns=['Subject no.', 'C value', 'Gamma value', 'scores_train', 'scores_test'])

cv = list(range(0,len(y),690))

ii = cv[subj]

test = list(range(ii, ii+690))
train = np.delete(list(range(0, len(y))), test, 0)

X_train=X[train]
y_train=y[train]

X_test=X[test]
y_test=y[test]

count = 0
classifiers = []

for gamma in gamma_2d_range:
    for C in C_2d_range:
        classifier = SVC(C=C, gamma=gamma, random_state=random_state)
        clf = classifier.fit(X_train, y_train)
        classifiers.append((C, gamma, clf))
        scores_train = clf.score(X_train, y_train)
        scores_test = clf.score(X_test, y_test)
        df.loc[count]=[subj+1, C, gamma, scores_train, scores_test]
        count += 1
        print(str(count))
            
#df.to_csv('LOSO_' + str(ii+1) + '_SVM_ASR.csv')

# pickle some variables
#filename = 'classifiers.pckl'
#pickle.dump(classifiers, open(filename, 'wb'))
