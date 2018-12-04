# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:23:18 2018

@author: Greta, grtu@dtu.dk

#EXAMPLE FUNCTION CALL
#	python runSVM.py -s 0

"""

#Imports
import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pickle 
import pandas as pd
import datetime

date = str(datetime.datetime.now())
date_str = date.replace(' ','-')
date_str = date_str.replace(':','.')
date_str = date_str[:-10]

# Load data
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

# Extract occipital electrodes
channels = list(range(0,len(X[1]),60))
channel_no = [1,6,7,8,20,21,32]

visual_idx = []
for no in channel_no:
    var = list(range(channels[no-1],(channels[no-1])+60))
    visual_idx.append(var)

flat_visual_idx = [item for sublist in visual_idx for item in sublist]

X_visual = X[:,flat_visual_idx]


y = ASR['Animate']
y = np.squeeze(y)

# Change y to 1 and -1
#y[y < 1] = -1
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)

print('============ Data Loaded ============')
print('X shape: ' + str(X_visual.shape))
print('y shape: ' + str(y.shape))


random_state = np.random.RandomState(0)

df = pd.DataFrame(columns=['Subject no.', 'scores_train', 'scores_test'])
cv = list(range(0,len(y),690))

count = 0

for counter, ii in enumerate(cv):
    
    classifier = SVC(random_state=random_state)
    test = list(range(ii, ii+690))
    train = np.delete(list(range(0, len(y))), test, 0)
    clf = classifier.fit(X_visual[train], y[train])
    scores_train = clf.score(X_visual[train], y[train])
    scores_test = clf.score(X_visual[test], y[test])
    df.loc[count]=[counter+1, scores_train, scores_test]
    count += 1
    print(str(count))

            
df.to_csv('noparameter_visual' + str(date_str) + '.csv')

# pickle some variables
#filename = 'classifiers.pckl'
#pickle.dump(classifiers, open(filename, 'wb'))
    

