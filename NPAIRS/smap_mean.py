#Imports
import numpy as np
from sklearn.svm import SVC
import scipy.io as sio
import pickle 
import pandas as pd
import datetime
import os

date = str(datetime.datetime.now())
date_str = date.replace(' ','-')
date_str = date_str.replace(':','.')
date_str = date_str[:-10]

# Load data
#os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)
y = y.astype(np.int16)
np.putmask(y, y<=0, -1)
y = y.astype(np.int16)

# Pick out 7 and 7 random subjects to fit a model on 

def makeSplit(eeg, labels):
    ''' Takes 7 and 7 random subjects from the EEG and labels variable.
    
    '''
    cv = list(range(0,len(y),690))
    subj_dict = dict(zip(cv,list(range(1,16))))
    fourteen = np.random.choice(cv, 14, replace = False)
    split1 = fourteen[0:7]
    split2 = fourteen[7:14]

    subs1 = []
    labels1 = []
    subs2 = []
    labels2 = []
    
    for count, ii in enumerate(split1):
        slice_idx = list(range(ii, ii+690))
        eeg = X[slice_idx]
        labels = y[slice_idx]
        subs1.append(eeg)
        labels1.append(labels)
        
        if count == 6:
            subs1 = np.concatenate(subs1,axis=0)
            labels1 = np.concatenate(labels1,axis=0)
            
    for count, ii in enumerate(split2):
        slice_idx = list(range(ii, ii+690))
        eeg = X[slice_idx]
        labels = y[slice_idx]
        subs2.append(eeg)
        labels2.append(labels)
        
        if count == 6:
            subs2 = np.concatenate(subs2,axis=0)
            labels2 = np.concatenate(labels2,axis=0)
            
    # Log subject splits
    lst1 = []
    for k in split1:
        s1 = subj_dict.get(k)
        lst1.append(s1)
        
    lst2 = []
    for k in split2:
        s2 = subj_dict.get(k)
        lst2.append(s2)
            
    return subs1, labels1, subs2, labels2, lst1, lst2

def runPairSVM():
    X1, y1, X2, y2, split1, split2 = makeSplit(X,y)

    # Log which subjects were used in the splits 
    os.chdir('SVM_pairs/')
    
    df = pd.DataFrame(columns=['Split 1 subjects', 'Split 2 subjects'])
    df.loc[1]=[split1, split2]
    df.to_excel('smap_split' + str(date_str) + '.xlsx')
    
    
    print('============ Data Loaded ============')
    print('Split 1, X1 shape: ' + str(X1.shape))
    print('Split 1, y1 shape: ' + str(y1.shape))
    print('Split 2, X2 shape: ' + str(X2.shape))
    print('Split 2, y2 shape: ' + str(y2.shape))
    print('Split 1, X1 shape: ' + str(X1.shape))
    print('Subjects split 1: ' + str(split1))
    print('Subjects split 2: ' + str(split2))
    
    random_state = np.random.RandomState(0)
    
    # Train SVM on 7 subjs, and 7 subjs again
    classifier1 = SVC(random_state=random_state, C=1.5, gamma=0.00005)
    clf1 = classifier1.fit(X1, y1)
    
    classifier2 = SVC(random_state=random_state, C=1.5, gamma=0.00005)
    clf2 = classifier2.fit(X2, y2)
    
    pickle.dump(clf1, open('SVM1_' + str(date_str) + '.pckl', 'wb'))
    pickle.dump(clf2, open('SVM2_' + str(date_str) + '.pckl', 'wb'))
    
    print('Saved two SVM classifiers, first one for subjects: ' + str(split1) + 'and second one for subjects: ' + str(split2))
  

