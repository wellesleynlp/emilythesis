# Emily Ahn
# 11.19.15
# Using SVC (Support Vector Classification)
from __future__ import division

from sklearn import svm
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import os
import sys
#from sklearn.metrics import confusion_matrix

if __name__=='__main__':
    """load .npz data, split into train-test folds, run training and testing"""
    npzdir = sys.argv[1] #'cslu_fae_corpus/npz' # directory with npz files

    langlist = ['AR', 'MA', 'HI']
    #langlist = ['HU', 'PO', 'RU']
    #langlist = ['JA', 'KO']
    #langlist = ['AR','BP','CA','CZ','FA','FR','GE','HI','HU','IN','IT','JA','KO','MA','MY','PO','PP','RU','SD','SP','SW','TA','VI']
    n_folds = 2
        
    accents_target = []
    data = {}

    for li, lang in enumerate(langlist):
        npzdata = np.load(os.path.join(npzdir, lang+'.npz'))   # load from npz file
        lang_data = np.array([npzdata[key] for key in npzdata.keys()])
        np.vstack(lang_data) #this potentially isn't doing anything
        data[lang] = lang_data
        accents_target.extend([li]*len(npzdata.keys()))
        
    #make sure arrays are numpy arrays!
    accents_target = np.array(accents_target)
    accents_data = np.array([data[lang] for lang in langlist])
    print "data len:", len(accents_data) # 3 (which should not be the case)
    #------ValueError: all the input array dimensions except for the concatenation axis must match exactly 
    np.vstack(accents_data)
    
    folds = StratifiedKFold(accents_target, n_folds = n_folds, shuffle = True)
    
    for foldid, (train_indices, test_indices) in enumerate(folds):

        X_train = np.array([accents_data[i] for i in train_indices])
        y_train = np.array([accents_target[i] for i in train_indices])
        X_test = np.array([accents_data[i] for i in test_indices])
        y_test = np.array([accents_target[i] for i in test_indices])
        print "X_train info:", X_train[0]
        
        # (diff. from GMM: no models to train)
        C = 1.0 # SVM regularization parameter
        svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        #--------When X_train is appropriately len of 742...
        #--------ValueError: setting an array element with a sequence. (below)
        svc.fit(X_train, y_train)
        #potential extra parameter for multi-class SVC: decision_function_shape='ovo'
        
        y_train_pred = svc.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        print 'Train accuracy for fold ', foldid, ': ', train_accuracy

        y_test_pred = svc.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        print 'Test accuracy for fold ', foldid, ': ', test_accuracy
        
        '''# now test
        accuracy = 0
        for test_index in test_indices:
            filename = filenames[test_index]
            actual_label = labels[test_index] # 0, 1, etc (lang index)
            actual_lang = langlist[actual_label]
            logprobs = {}   # dict: total log prob of this file under each model
            for lang in langlist:
                logprobs[lang] = apply_model(models[lang], data[actual_lang][filename])
            predicted_lang = max(logprobs.items(), key=lambda x:x[1])[0]
            # insert prediction (of lang index) into predicted list
            predicted_list.append(predicted_lang)
            actual_list.append(actual_lang)
            if actual_lang == predicted_lang:
                accuracy += 1
        accuracy_list[foldid] = accuracy/len(test_indices)
        print 'Accuracy for fold', foldid, 'is', accuracy_list[foldid]

    print 'AVG over', len(folds), 'folds is', np.average(accuracy_list)'''
    
    '''#CONFUSION MATRIX (y_test, y_pred) -> (actual label, predictions)
    cm = confusion_matrix(actual_list, predicted_list, labels=langlist) 
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)'''