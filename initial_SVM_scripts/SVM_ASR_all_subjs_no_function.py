
# Notebook with SVM code - cross validation of C value in RBF kernel SVM, animate vs inanimate with ASR loaded data
# Greta, 25/09/2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.svm import SVC
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import os
import sys
import pandas as pd
import scipy.io as sio
import pickle

print(df)

os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')

# LOAD EEG DATA #
ASR = sio.loadmat('ASRnew')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

####### FOR SVM #######
cv = list(range(0,len(y),690))
random_state = np.random.RandomState(0)
gamma_val =   1
df = pd.DataFrame(columns=['Subject no.', 'C value', 'scores_train', 'scores_test'])

C_range = [0.1, 1]

count = 0
for counter, ii in enumerate(cv):
    for C in C_range:
        classifier = svm.SVC(C=C, random_state=random_state, gamma=gamma_val)
        test = list(range(ii, ii+690))
        train = np.delete(list(range(0, len(y))), test, 0)
        clf = classifier.fit(X[train], y[train])
        scores_train = clf.score(X[train], y[train])
        scores_test = clf.score(X[test], y[test])
        df.loc[count]=[counter+1, C, scores_train, scores_test]
        count += 1
        print(str(count))
        
dualcoef = clf.dual_coef_
supvectors = clf.support_vectors_


filename = 'dualcoef.pckl'
filename2 = 'support_vectors.pckl'
pickle.dump(dualcoef, open(filename, 'wb'))
pickle.dump(supvectors, open(filename2, 'wb'))

pkl_filename = 'SVM_model.pckl'
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file)


df.to_csv('../data/svm_training_LOSO_RBF_ASR_savemodel.csv')


print('============ Data Loaded ============')
print('X shape: ' + str(X.shape))
print('y shape: ' + str(y.shape))
