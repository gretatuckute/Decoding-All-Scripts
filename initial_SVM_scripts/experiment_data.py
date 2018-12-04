from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import scipy.io as sio
from scipy import signal
import sys
import numpy as np
import pandas as pd
import os
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
#sys.path.append(".../")  # go to parent dir
#from mrcode.utils.file_utils import folderFinder
#from mrcode import settings

# To make imports work on Greta's PC uncomment following:
os.chdir('C:/Users/Greta/Documents/GitHub/Project-MindReading/mrcode/')
from utils.file_utils import folderFinder
import settings


def std_windows(eeg, time_window=50):
    temp_eeg = []
    for ii in range(eeg.shape[0]):
        temp_epoch = eeg[ii, :, :]
        temp_eeg.append(
            np.std(temp_epoch.reshape(int(temp_epoch.shape[0] / time_window), time_window, temp_epoch.shape[1]),
                   1).flatten())
    std_eeg = np.array(temp_eeg)
    return std_eeg


def abs_features(eeg):
    abs_eeg = abs(np.reshape(eeg, [eeg.shape[0], -1]))
    return abs_eeg


def std_overlap_windows(eeg, time_window=80, overlap=0.5):
    temp_eeg = []
    for ii in range(eeg.shape[0]):
        temp_epoch = eeg[ii, :, :]
        std_channels = []
        for jj in range(temp_epoch.shape[1]):
            channel = temp_epoch[:, jj]

            std_channel = []
            for kk in range(int(time_window * overlap)):
                std_channel.append(channel[kk:(kk + time_window)])

            std_channels.append(np.std(std_channel, 1))

        temp_eeg.append(std_channels)

    temp_eeg_array = np.array(temp_eeg)
    std_temp_eeg = temp_eeg_array.reshape(temp_eeg_array.shape[0], temp_eeg_array.shape[2] * temp_eeg_array.shape[1])
    return std_temp_eeg


def load_data_generator(folders, load_mode='raw', target='category'):
    ''' This function sorts the EEG and categories based on a categories and image_id, thus in
        the exact same manner across subjects. The sorting is alphabetical.
    '''
    for ii in folders:
        data = sio.loadmat(ii + '/eeg_events.mat')
        eeg = data['eeg_events'].transpose()

        if load_mode == 'raw':
            None

        if load_mode == 'raw_one_feature':
            eeg = eeg.reshape(eeg.shape[0], -1)

        if load_mode == 'std':
            eeg = std_windows(eeg)

        if load_mode == 'std_overlap':
            eeg = std_overlap_windows(eeg)

        if load_mode == 'abs_features':
            eeg = abs_features(eeg)

        image_info = pd.read_csv(ii + '/image_order.txt', delimiter='\t')
        sorted_image_info = image_info.sort_values(['category', 'image_id']) # Sort image info based on category and image_id        
        # new_sorted_image_info = image_info.sort_values(['supercategory','image_id'])
        # categories = sorted_image_info[target].as_matrix() #Can check how the categories look like
        
        sort_idx = list(sorted_image_info.index.astype(int)) # Indices of the sorted image info
        
        eeg = eeg[sort_idx] # Sorting EEG in the same manner as image info

        # Use code below if running on sorted EEG, and categories
        
        if target == 'animate':
            categories = []
            temp_categories = list(sorted_image_info['supercategory'].as_matrix())
            for category in temp_categories:
                if category != 'animal':
                    categories.append('inanimate')
                else:
                    categories.append('animate')
        else:
            categories = list(sorted_image_info[target].as_matrix()) # 0 is inanimate, 1 is animate
        yield eeg, categories

def load_data_generator_animate(folders, load_mode='raw', target='animate'):
    ''' This function sorts the EEG and categories based on a manually defined order of categories 
        in cat_lst. The images are NOT sorted based on image_id, but simply the categories.
        The EEG is sorted based on the same manner in which the images are sorted.
    '''
    for ii in folders:
        data = sio.loadmat(ii + '/eeg_events.mat')
        eeg = data['eeg_events'].transpose()

        if load_mode == 'raw':
            None

        if load_mode == 'raw_one_feature':
            eeg = eeg.reshape(eeg.shape[0], -1)

        if load_mode == 'std':
            eeg = std_windows(eeg)

        if load_mode == 'std_overlap':
            eeg = std_overlap_windows(eeg)

        if load_mode == 'abs_features':
            eeg = abs_features(eeg)

        image_info = pd.read_csv(ii + '/image_order.txt', delimiter='\t')
        # new_sorted_image_info = image_info.sort_values(['supercategory','image_id'])
        
        cat_lst = (['dog']*30)+(['cat']*30)+(['horse']*30)+(['cow']*30)+(['sheep']*30)+(['giraffe']*30)+(['elephant']*30)+(['zebra']*30)+(['bear']*30)+(['bird']*30)+(['teddy bear']*30)+(['airplane']*30)+(['boat']*30)+(['motorcycle']*30)+(['bus']*30)+(['train']*30)+(['stop sign']*30)+(['clock']*30)+(['bench']*30)+(['bed']*30)+(['toilet']*30)+(['donut']*30)+(['pizza']*30)
        cat_lst_unique = sorted(set(cat_lst), key = cat_lst.index)
        
        animate_idx = [] # List of indices of with categories in order as cat_lst_unique
        
        for cat in cat_lst_unique:
            for index, row in image_info.iterrows():
                if row['category'] == cat:
                    animate_idx.append(index)
                    
        eeg = eeg[animate_idx] # Sorting EEG in the same manner as image info
        
        categories = []
        if target == 'animate':
            categories = (['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['animate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)+(['inanimate']*30)
            
        yield eeg, categories

# def smote_sampling(y, y_all, X_before_padding, sample_percentage=0.035):
#     # enabling SMOTE sampling
#     from_categorical_dict = dict(enumerate(y.columns))
#     to_categorical_dict = {v: k for k, v in from_categorical_dict.items()}
#     y_categorical = np.array([to_categorical_dict[ii] for ii in y_all])
#
#     class_images = len(X_before_padding)/23
#     new_class_images = class_images + class_images*sample_percentage
#
#     X_train_padding = np.zeros((new_class_images, X_before_padding.shape[1]))
#     y_train_padding = np.ones((new_class_images), int) * len(y.columns)
#     X_with_padding = np.vstack([X_before_padding, X_train_padding])
#     y_with_padding = np.concatenate([y_categorical, y_train_padding])
#
#     # Use SMOTE to generate new samples equal to the amount used in the test set
#     sm = SMOTE(ratio='all', random_state=1)
#     X_res, y_res = sm.fit_sample(X_with_padding, y_with_padding)
#
#     # Remove padding from training set
#     padding_to_remove = np.where(~X_res.any(axis=1))[0]
#     X_train = np.delete(X_res, padding_to_remove, 0)
#     y_train_categorical = np.delete(y_res, padding_to_remove, 0)
#
#     y_train_temp2 = np.array([from_categorical_dict[ii] for ii in y_train_categorical])
#     y2 = pd.get_dummies(y_train_temp2)
#     y_train = np.array(y2)
#
#     return X_train, y_train


def load_data(target='category', load_mode='raw_one_feature', cnn_ready=False, apply_smote=False):
    experiment_data_path = settings.experiment_data_init()
    experiment_folders = [experiment_data_path + '/' + ii for ii in folderFinder(experiment_data_path)]

    X = []
    y = []
    for eeg, categories in load_data_generator(experiment_folders, load_mode=load_mode, target=target):
        X.extend(eeg)
        y.extend(categories)

    X = np.array(X)
    y = np.array(y)
    y_dummies = pd.get_dummies(y)
    y = np.array(y_dummies)

    add_lst = list(range(0, len(X), 690))

    test_cat = list(range(60, 90))  # Taking out category for test set
    b = []
    for i in add_lst:
        for j in test_cat:
            k = j + i
            b.append(k)
            
    val_cat = list(range(630, 660))  # Taking out category for validation
    c = []
    for ii in add_lst:
        for jj in val_cat:
            kk = jj + ii
            c.append(kk)

    d = b + c # To delete both test and validation sets from y and X
    
    y_test, y_val, y_train = y[b], y[c], np.delete(y, d, 0) #delete both b and c from the train set
    X_test, X_val, X_train = X[b], X[c], np.delete(X, d, 0)

    # if apply_smote:
    #     X_train, y_train = smote_sampling(y_dummies, y, X)

    # b = list(range(10, 690 * len(experiment_folders), 15)) # Starts at 10 and takes every 15th element, i.e. 10, 25, 40...690
    
    # If to use an entire category (30 images/ERPs) as a testset and another category as validation:
    # Make a list, b, that randomly takes out 2 categories
    # The data is currently sorted in the same manner across subjects, category-wise?

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.066, random_state=42)

    # Split data
#    y_test, y_train = y[b], np.delete(y, b, 0) # y_test are the values from b, e.g. 690. y_train deletes b from y
#    X_test, X_train = X[b], np.delete(X, b, 0)
#
#    y_test, y_val = y_test[::2], y_test[1::2] # y test is now 690 length. every second goes to test, and every second to validation
#    X_test, X_val = X_test[::2], X_test[1::2]

    if load_mode == 'raw':
        # Zero mean data
        for ii in range(X_train.shape[2]):
            chan_mean = np.mean(X_train[:, :, ii])
            X_test[:, :, ii] -= chan_mean
            X_val[:, :, ii] -= chan_mean
            X_train[:, :, ii] -= chan_mean

    else:
        # Scale data
        xScale = RobustScaler().fit(X_train)
        X_train = xScale.transform(X_train)
        X_test = xScale.transform(X_test)
        X_val = xScale.transform(X_val)

    # Expand dim for CNN training
    if cnn_ready and load_mode == 'raw':
        X_train, X_test, X_val = [np.expand_dims(ii, axis=3) for ii in [X_train, X_test, X_val]]
        y_train, y_test, y_val = [np.expand_dims(ii, axis=1) for ii in [y_train, y_test, y_val]]

    if cnn_ready and load_mode != 'raw':
        X_train, X_test, X_val = [np.expand_dims(ii, axis=1) for ii in [X_train, X_test, X_val]]
        y_train, y_test, y_val = [np.expand_dims(ii, axis=1) for ii in [y_train, y_test, y_val]]

    # Printing shape of data
    print('============ Data Loaded ============')
    print('X_train shape: ' + str(X_train.shape))
    print('X_val shape: ' + str(X_val.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('y_val shape: ' + str(y_val.shape))
    print('y_test shape: ' + str(y_test.shape) + '\n\n')
    print('categories: ' + str(categories))

    return X_train, y_train, X_test, y_test, X_val, y_val


def load_data_svm(target='category', load_mode='raw_one_feature'):
    experiment_data_path = settings.experiment_data_init()
    experiment_folders = [experiment_data_path + '/' + ii for ii in folderFinder(experiment_data_path)]

    X = []
    y = []
    
    # IF NOT SORTED IN ANIMATE, ANIMATE..... INANIMATE, then do load_data_generator instead!
    
    for eeg, categories in load_data_generator_animate(experiment_folders, load_mode=load_mode, target=target):
        X.extend(eeg)
        y.extend(categories)

    X = np.array(X)
    y = np.array(y)
    y_dummies = pd.get_dummies(y)
    y = np.array(y_dummies)

    if load_mode == 'raw':
        # Zero mean data
        for ii in range(X.shape[2]):
            chan_mean = np.mean(X[:, :, ii])
            X[:, :, ii] -= chan_mean
        # Trying to create raw_one_feature based on the zero meaned raw EEG
        X = X.reshape(X.shape[0], -1)
        X = signal.resample(X,3520,axis=1) # Resampling
        
        # Try additional scaling besides zero meaning?
        xScale = RobustScaler().fit(X)
        X = xScale.transform(X)
        
    else:
        # Scale data
        xScale = RobustScaler().fit(X)
        X = xScale.transform(X)
        
    # RESAMPLE EEG if load_mode = raw_one_feature
    if load_mode == 'raw_one_feature':
        X = signal.resample(X,3520,axis=1) # 1/5 samples
    
    y1 = []
    for ii in y: # Creating a one dimensional y (adding 0 and 1 in the same vector)
        if ii[0] == 1:
            y1.append(0)
        if ii[1] == 1:
            y1.append(1)

    y1 = np.array(y1)
        
#    if target == 'category':
#         y1 = y

    # Printing shape of data
    print('============ Data Loaded ============')
    print('X_train shape: ' + str(X.shape))
    print('y_train shape: ' + str(y1.shape))
    print(y1)
    # print('categories: ' + str(categories))

    return X, y1

def load_data_svm_cat(target='category', load_mode='raw_one_feature'):
    '''
    Loads data to SVMs, with one dimensional y, and takes out the same category across subjects for test set
    '''
    experiment_data_path = settings.experiment_data_init()
    experiment_folders = [experiment_data_path + '/' + ii for ii in folderFinder(experiment_data_path)]

    X = []
    y = []
    for eeg, categories in load_data_generator(experiment_folders, load_mode=load_mode, target=target):
        X.extend(eeg)
        y.extend(categories)

    X = np.array(X)
    y = np.array(y)
    y_dummies = pd.get_dummies(y)
    y = np.array(y_dummies)

    y1 = []
    
    for ii in y:  # Creating a one dimensional y (adding 0 and 1 in the same vector)
        for jj in list(range(0,23)):
            if ii[jj] == 1:
                y1.append(jj+1)
            
    y1 = np.array(y1)

    if load_mode == 'raw':
        # Zero mean data
        for ii in range(X.shape[2]):
            chan_mean = np.mean(X[:, :, ii])
            X[:, :, ii] -= chan_mean

    else:
        # Scale data
        xScale = RobustScaler().fit(X)
        X = xScale.transform(X)
        
    add_lst = list(range(0, len(X), 690))

    test_cat = list(range(60, 90))  # Taking out category for test set
    b = []
    for i in add_lst:
        for j in test_cat:
            k = j + i
            b.append(k)
                
    y_test, y_train = y1[b], np.delete(y1, b, 0) #delete both b from the train set
    X_test, X_train = X[b], np.delete(X, b, 0)

    # Printing shape of data
    print('X_train shape: ' + str(X_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('y_test shape: ' + str(y_test.shape))
    print('categories: ' + str(categories))

    return X_train, y_train, X_test, y_test


def load_data_NN_LOSO(target='animate', load_mode='raw_one_feature',LOSO=9, SVM=False):
    '''
    Leave one subject out approach, can be used to load data into NN's or SVMs (if SVM=True)
    '''
    
    experiment_data_path = settings.experiment_data_init()
    experiment_folders = [experiment_data_path + '/' + ii for ii in folderFinder(experiment_data_path)]

    X = []
    y = []
    for eeg, categories in load_data_generator(experiment_folders, load_mode=load_mode, target=target):
        X.extend(eeg)
        y.extend(categories)

    X = np.array(X)
    y = np.array(y)
    y_dummies = pd.get_dummies(y)
    y = np.array(y_dummies)

    if load_mode == 'raw':
        # Zero mean data
        for ii in range(X.shape[2]):
            chan_mean = np.mean(X[:, :, ii])
            X[:, :, ii] -= chan_mean

    else:
        # Scale data
        xScale = RobustScaler().fit(X)
        X = xScale.transform(X)
        
    if SVM == True: #Can flatten y into a 1 dimensional vector
        
        y1 = []
        for ii in y: # Creating a one dimensional y (adding 0 and 1 in the same vector)
            if ii[0] == 1:
                y1.append(0)
            if ii[1] == 1:
                y1.append(1)
    
        y = np.array(y1)
        
    
    # Take out a single subject. The LOSO arg specifies which one to take out (if LOSO=0, takes out 1st sub)    
    test = list(range(LOSO, LOSO+690))
    train = np.delete(list(range(0, len(y))), test, 0)
    
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]


    # Printing shape of data
    print('============ Data Loaded ============')
    print('X_train shape: ' + str(X_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('y_test shape: ' + str(y_test.shape))
    print('categories: ' + str(categories))

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # X_train1, y_train1, X_test1, y_test1, X_val1, y_val1 = load_data(load_mode='abs_features', apply_smote=False, cnn_ready=False)
    # X_train1, y_train1, X_test1, y_test1 = load_data(load_mode='raw_one_feature', apply_smote=False, cnn_ready=True)
    X, y1 = load_data_svm(target='category', load_mode='std_overlap')
    print('hej')

