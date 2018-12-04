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
#os.chdir('C:/Users/Greta/Documents/GitHub/decoding/data/ASR/')
ASR = sio.loadmat('ASRinterp')
X = ASR['EV_Z']

y = ASR['Animate']
y = np.squeeze(y)

print('============ Data Loaded ============')
print('X shape: ' + str(X.shape))
print('y shape: ' + str(y.shape))


random_state = np.random.RandomState(0)

# Training on all subjects.

classifier = SVC(random_state=random_state, C=1.5, gamma=0.00005)
clf = classifier.fit(X, y)

dualcoef = clf.dual_coef_
supvectors = clf.support_vectors_

filename = 'dualcoef_mean_parameter.pckl'
filename2 = 'support_vectors_mean_parameter.pckl'
pickle.dump(dualcoef, open(filename, 'wb'))
pickle.dump(supvectors, open(filename2, 'wb'))

pkl_filename = 'SVM_model_mean_parameter.pckl'
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file)
    

