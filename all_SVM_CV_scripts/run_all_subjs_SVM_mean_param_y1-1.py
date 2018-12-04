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

print('============ Data Loaded ============')
print('X shape: ' + str(X.shape))
print('y shape: ' + str(y.shape))
print('y: ' + str(y[0:40]))


random_state = np.random.RandomState(0)

df = pd.DataFrame(columns=['Subject no.', 'scores_train', 'scores_test'])
cv = list(range(0,len(y),690))

count = 0

for counter, ii in enumerate(cv):
    
    classifier = SVC(random_state=random_state,C=1.5, gamma=0.00005)
    test = list(range(ii, ii+690))
    train = np.delete(list(range(0, len(y))), test, 0)
    clf = classifier.fit(X[train], y[train])
    scores_train = clf.score(X[train], y[train])
    scores_test = clf.score(X[test], y[test])
    df.loc[count]=[counter+1, scores_train, scores_test]
    count += 1
    print(str(count))
    pkl_filename = 'SVM_model_y1-1_REALmean_parameter_subj' + str(count+1) + '.pckl'
    pickle.dump(clf, open(pkl_filename, 'wb'))

            
df.to_csv('LOSO_all_subjs_mean_parameter_y1-1_' + str(date_str) + '.csv')

# pickle some variables
#filename = 'classifiers.pckl'
#pickle.dump(classifiers, open(filename, 'wb'))
    

