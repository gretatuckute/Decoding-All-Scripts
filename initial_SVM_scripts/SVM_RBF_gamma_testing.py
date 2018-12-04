
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

from scipy.spatial import distance
os.chdir('C:/Users/Greta/Documents/GitHub/Project-MindReading/mrcode/')

# sys.path.append('../')  # go to parent dir
from mrcode.preprocessing import experiment_data

X, y = experiment_data.load_data_svm(load_mode='raw',target='animate')
# if using load_data_svm the data is sorted in animate/inanimate
# load_data_svm calls load_data_generator_animate

######### Matrices and gamma SVM evaluation ##############

M = distance.pdist(X,'euclidean')
Msquare = distance.squareform(M)
Mexp = np.exp(-M**2)
Mexpsquare = distance.squareform(Mexp)

Mexp_gamma = np.exp((np.square(1/100))*(-(np.square(M))))
Mexpsquare_gamma = distance.squareform(Mexp_gamma)

plt.matshow(Mexpsquare_gamma)

a=np.amax(X)
b=np.amin(X)
c=np.mean(X)

d=np.amax(M)
e=np.amin(M)
f=np.mean(M)

plt.matshow(Mexpsquare)

# Plot with all animate categories sorted first (not dependent on subject)
add_lst = list(range(0, len(X), 690))

animate_cats = list(range(0, 300)) # Taking out all animate categories
inanimate_cats = list(range(300,690))

all_animate = []
for i in add_lst:
    for j in animate_cats:
        k = j + i
        all_animate.append(k)
        
all_inanimate = []
for i in add_lst:
    for j in inanimate_cats:
        k = j + i
        all_inanimate.append(k)


X_sort = np.vstack((X[all_animate],X[all_inanimate]))

M_sort = distance.pdist(X_sort,'euclidean')
Mexp_sort = np.exp(-M_sort**2)
Mexpsquare_sort = distance.squareform(Mexp_sort)

Mexp_gamma_sort = np.exp((np.square(1/100))*(-(np.square(M_sort))))
Mexpsquare_gamma_sort = distance.squareform(Mexp_gamma_sort)

plt.matshow(Mexpsquare_gamma_sort)

# SORT CUTE FIRST #
cute_cats = list(range(0, 90)) # Taking out all animate categories
ugly_cats = list(range(90,690))

all_cute = []
for i in add_lst:
    for j in cute_cats:
        k = j + i
        all_cute.append(k)
        
all_ugly = []
for i in add_lst:
    for j in ugly_cats:
        k = j + i
        all_ugly.append(k)
        
X_cute = np.vstack((X[all_cute],X[all_ugly]))

M_cute = distance.pdist(X_cute,'euclidean')
Mexp_cute = np.exp(-M_cute**2)
Mexpsquare_cute = distance.squareform(Mexp_cute)

Mexp_gamma_cute = np.exp((np.square(1/100))*(-(np.square(M_cute))))
Mexpsquare_gamma_cute = distance.squareform(Mexp_gamma_cute)

plt.matshow(Mexpsquare_gamma_cute)


# Loading std_overlap
X, y = experiment_data.load_data_svm(load_mode='std_overlap',target='animate')
# OBS: values too low to show Mexpsquare.

plt.hist(M,bins=range(1, 1000))

# Insert NaN values in M matrix if value is above ... 

M_array = np.asarray(M)
M_array[M_array > 300] = np.nan #OBS overwriting M_array

# Compute the exp
Mexp_nan = np.exp(-(np.square(M_array)))
Mexpsquare_nan = distance.squareform(Mexp_nan)

Mexp_nan_gamma = np.exp((np.square(1/50))*(-(np.square(M_array))))
Mexpsquare_nan_gamma = distance.squareform(Mexp_nan_gamma)

plt.matshow(Mexpsquare_nan_gamma)


sqtest = np.square(M_array)

####### FOR SVM #######
cv = list(range(0,len(y),690))
random_state = np.random.RandomState(0)  
df = pd.DataFrame(columns=['Subject no.', 'C value', 'scores_train', 'scores_test'])

C_range = [0.001, 0.01, 0.1, 1, 10]

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

df.to_csv('../data/svm_training_LOSO_RBF_kernel_std_overlap.csv')
