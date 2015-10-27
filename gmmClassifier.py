from __future__ import division

"""Accent classification using separate GMMs for each accent. 
See http://www.ece.mcgill.ca/~rrose1/papers/reynolds_rose_sap95.pdf for outline of process.
"""

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM
import os
import sys

def train_model(speech_array_list, n_components):
    """train GMM with n_components mixtures from data,
    represented as a list of arrays, where each array contains the
    PLPs from a speech file. Returns this model."""
    # to do
    return 

def apply_model(gmm_model, speech_array):
    """compute total log probability (sum across all rows) of
    speech_array under this model"""
    # to do
    return 0

if __name__=='__main__':
    """load .npz data, split into train-test folds, run training and testing"""
    npzdir = sys.argv[1]  # directory with npz files
    n_components = sys.argv[2]  # number of GMM components

    langlist = ['AR', 'MA', 'HI']
    n_folds = 10

    data = {}
    filenames = []  # list of length = number of files. nth element is filename of nth file.
    labels = []   # list of length = number of files. nth element is integer corresponding to the nth file's accent
    
    for li, lang in enumerate(langlist):
        with np.load(os.path.join(npzdir, lang+'.npz')) as x:
            data[lang] = dict(x)
        # format: data['AR'] = {'FAR00001': speech PLP array}
        
        filenames.extend(data[lang].keys())
        labels.extend([li]*len(data[lang]))
    
    folds = StratifiedKFold(labels, n_folds = n_folds, shuffle = True)
    for foldid, (train_indices, test_indices) in enumerate(folds):
        models = {}   # store models for each lang
        for li, lang in enumerate(langlist):
            
            train_lang_indices = [i for i in train_indices if labels[i] == li]
            # indices corresponding to this lang

            train_lang_list = [data[lang][filenames[i]] \
                                   for i in train_lang_indices]
            # list of PLP arrays corresponding to filenames in training
                        
            models[lang] = train_model(train_lang_list, n_components)
        
        # now test
        accuracy = 0
        for test_index in test_indices:
            filename = filenames[test_index]
            actual_label = labels[test_index]
            actual_lang = langlist[actual_label]
            logprobs = {}   # total log prob of this file under each model
            for lang in langlist:
                logprobs[lang] = apply_model(models[lang], data[actual_lang][filename])
            predicted_lang = max(logprobs.items(), key=lambda x:x[1])[0]
            if actual_lang == predicted_lang:
                accuracy += 1
        
        print 'Accuracy for fold', foldid, 'is', accuracy/len(test_indices)


            
