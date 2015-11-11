# Emily Ahn
# 11.11.15
from __future__ import division

"""Accent classification using separate GMMs for each accent. 
See http://www.ece.mcgill.ca/~rrose1/papers/reynolds_rose_sap95.pdf for outline of process.
"""

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM
import os
import sys
from sklearn.metrics import confusion_matrix

def train_model(speech_array_list, n_components, covar_type): 
    """train GMM with n_components mixtures from data,
    represented as a list of arrays, where each array contains the
    PLPs from a speech file. Returns this model."""

    bigArray = np.vstack(speech_array_list)
    print "bigArray DONE"
    g = GMM(n_components=n_components,covariance_type=covar_type, init_params='wc', n_iter=20)
    g.fit(bigArray)
    #print "fitting DONE"
    return g

def apply_model(gmm_model, speech_array):
    # given 1 speaker (1 file), figure out
    """compute total log probability (sum across all rows) of
    speech_array under this model"""
    # assume that each time stamp's log-probs are INDEPENDENT
    return np.sum(gmm_model.score(speech_array))

if __name__=='__main__':
    """load .npz data, split into train-test folds, run training and testing"""
    npzdir = sys.argv[1] #'cslu_fae_corpus/npz' # directory with npz files
    n_components = int(sys.argv[2])  # number of GMM components
    covar = sys.argv[3]    # covar types: ['spherical', 'diag', 'tied', 'full']

    #langlist = ['AR', 'MA', 'HI']
    #langlist = ['HU', 'PO', 'RU']
    #langlist = ['JA', 'KO']
    langlist = ['AR','BP','CA','CZ','FA','FR','GE','HI','HU','IN','IT','JA','KO','MA','MY','PO','PP','RU','SD','SP','SW','TA','VI']
    n_folds = 4

    data = {}
    filenames = []  # list of length = number of files. nth element is filename of nth file.
    labels = []   # list of length = number of files. nth element is integer corresponding to the nth file's accent
    
    # for each Lang, create dictionary "data"
    for li, lang in enumerate(langlist):
        with np.load(os.path.join(npzdir, lang+'.npz')) as x:
            data[lang] = dict(x)
        # format: data['AR'] = {'FAR00001': speech PLP array}
        
        filenames.extend(data[lang].keys())
        labels.extend([li]*len(data[lang]))
    
    folds = StratifiedKFold(labels, n_folds = n_folds, shuffle = True)
    # initialize list to put accuracy of each fold, to be averaged over & printed
    accuracy_list = np.empty(len(folds))
    
    # initialize lists to put in predictions & results. For Confusion matrix
    predicted_list = []
    actual_list = []
    
    for foldid, (train_indices, test_indices) in enumerate(folds):
        models = {}   # store models for each lang
        for li, lang in enumerate(langlist):
            
            train_lang_indices = [i for i in train_indices if labels[i] == li]
            # indices corresponding to this lang

            train_lang_list = [data[lang][filenames[i]] \
                                   for i in train_lang_indices]
            # list of PLP arrays corresponding to filenames in training
                        
            models[lang] = train_model(train_lang_list, n_components, covar)
        
        # now test
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

    print 'AVG over', len(folds), 'folds is', np.average(accuracy_list)
    
    #CONFUSION MATRIX (y_test, y_pred) -> (actual label, predictions)
    cm = confusion_matrix(actual_list, predicted_list, labels=langlist) 
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
