from __future__ import division

__author__='Emily Ahn and Sravana Reddy'

""" Date created: 10/29/2015
    Date modified: 11/14/2015
    ****************************
    Accent classification using separate GMMs for each accent. 
    See http://www.ece.mcgill.ca/~rrose1/papers/reynolds_rose_sap95.pdf for outline of process.
"""

import numpy as np
from sklearn.mixture import GMM
import os
import sys
from sklearn.metrics import confusion_matrix

def train_model(speech_array_list, n_components, covar_type): 
    """train GMM with n_components mixtures from data,
    represented as a list of arrays, where each array contains the
    PLPs from a speech file. Returns this model."""

    bigArray = np.vstack(speech_array_list)
    g = GMM(n_components=n_components,covariance_type=covar_type)
    g.fit(bigArray)
    return g

def apply_model(gmm_model, speech_array):
    # given 1 speaker (1 file), figure out
    """compute total log probability (sum across all rows) of
    speech_array under this model"""
    # assume that each time stamp's log-probs are INDEPENDENT
    return np.sum(gmm_model.score(speech_array))

def load_data(npzdir, langlist):
    data = {}
    # for each Lang, create dictionary in "data"                                                     
    for li, lang in enumerate(langlist):
        with np.load(os.path.join(npzdir, lang+'.npz')) as x:
            data[lang] = dict(x)
        print 'Loaded compressed data for', lang
    return data

def get_train_data(data, lang):
    """return list of PLP arrays corresponding to filenames in training for this lang"""
    train_files = open(os.path.join('traintestsplit', lang+'.trainlist')).read().split()
    return [data[lang][filename+'.npytxt'] for filename in train_files]

def run_test(models, data):
    """apply trained models to each file in test data"""
    num_correct = 0.0
    num_total = 0.0

    # initialize lists to put in predictions & results for confusion matrix 
    predicted_labels = []
    actual_labels = []
    langlist = models.keys()
    for ai, actual_lang in enumerate(langlist):
        test_files = open(os.path.join('traintestsplit', actual_lang+'.testlist')).read().split()
        for filename in test_files:
            logprobs = {}   # dict: total log prob of this file under each model                    
            for test_lang in langlist:
                logprobs[test_lang] = apply_model(models[test_lang], data[actual_lang][filename+'.npytxt'])
            predicted_lang = max(logprobs.items(), key=lambda x:x[1])[0]
            # insert prediction (of lang index) into predicted list                                     
            predicted_labels.append(langlist.index(predicted_lang))
            actual_labels.append(ai)
            if actual_lang == predicted_lang:
                num_correct += 1
            num_total += 1

    print
    print 'Accuracy', num_correct*100/num_total

    #CONFUSION MATRIX (y_test, y_pred) -> (actual label, predictions)                                   
    cm = confusion_matrix(actual_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # display confusion stats by lang (TODO: visualize with matplotlib)                                 
    print '*'*20
    for ai, actual_lang in enumerate(langlist):
        print actual_lang, 'confusion:'
        for pi, predicted_lang in enumerate(langlist):
            print '{0}: {1:.2f}%'.format(predicted_lang, cm_normalized[ai, pi]*100)
        print '*'*20

if __name__=='__main__':
    """load .npz data, split into train-test folds, run training and testing"""
    npzdir = sys.argv[1] #'cslu_fae_corpus/npz' # directory with npz files
    n_components = int(sys.argv[2])  # number of GMM components
    covar = sys.argv[3]    # covar types: ['spherical', 'diag', 'tied', 'full']

    #langlist = ['AR', 'MA', 'HI']
    #langlist = ['HU', 'PO', 'RU']
    #langlist = ['JA', 'KO']
    #langlist = ['AR','BP','CA','CZ','FA','FR','GE','HI','HU','IN','IT','JA','KO','MA','MY','PO','PP','RU','SD','SP','SW','TA','VI']
    langlist = ['AR', 'CZ', 'IN']

    data = load_data(npzdir, langlist)
    
    models = {}   # store trained models for each lang
    for li, lang in enumerate(langlist):
        train_lang_list = get_train_data(data, lang)
        models[lang] = train_model(train_lang_list, n_components, covar)
        print 'Trained model for', lang
    
    # now test
    run_test(models, data)
