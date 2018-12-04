
# Notebook with SVM code - cross validation of C value in RBF kernel SVM, animate vs inanimate
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

sys.path.append('../')  # go to parent dir
from mrcode.preprocessing import experiment_data

X, y = experiment_data.load_data_svm(load_mode='raw_one_feature')

cv = list(range(0,len(y),690))
random_state = np.random.RandomState(0)  
df = pd.DataFrame(columns=['Subject no.', 'C value', 'scores_train', 'scores_test'])

C_range = [0.5, 1, 5]

count = 0
for counter, ii in enumerate(cv):
    for C in C_range:
        classifier = svm.SVC(C=C, random_state=random_state)
        test = list(range(ii, ii+690))
        train = np.delete(list(range(0, len(y))), test, 0)
        clf = classifier.fit(X[train], y[train])
        scores_train = clf.score(X[train], y[train])
        scores_test = clf.score(X[test], y[test])
        df.loc[count]=[counter+1, C, scores_train, scores_test]
        count += 1
        print(str(count))

df.to_csv('../data/svm_training_LOSO_RBF_kernel_raw.csv')
