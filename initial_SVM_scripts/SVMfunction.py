# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:02:03 2018

@author: Greta
"""

def SVMfunction(subjID):
    '''
    Argument: subject number to leave out as test set.
    
    Runs cross-validation on C and gamma parameters on RBF kernel SVM
    
    '''
    
    

    C_2d_range = [0.1, 1.0, 10.0, 100.0]
    gamma_2d_range = [0.05, 0.005, 0.0005, 0.00005]
    
    random_state = np.random.RandomState(0)
    
    df = pd.DataFrame(columns=['Subject no.', 'C value', 'Gamma value', 'scores_train', 'scores_test'])
    cv = list(range(0,len(y),690))
    
    count = 0
    classifiers = []
    for counter, ii in enumerate(cv):
        for gamma in gamma_2d_range:
            for C in C_2d_range:
                classifier = SVC(C=C, gamma=gamma, random_state=random_state)
                test = list(range(ii, ii+690))
                train = np.delete(list(range(0, len(y))), test, 0)
                clf = classifier.fit(X[train], y[train])
                classifiers.append((C, gamma, clf))
                scores_train = clf.score(X[train], y[train])
                scores_test = clf.score(X[test], y[test])
                df.loc[count]=[counter+1, C, gamma, scores_train, scores_test]
                count += 1
                print(str(count))
                
    df.to_csv('LOSO_' + str(subjID) + '_SVM_ASR.csv')