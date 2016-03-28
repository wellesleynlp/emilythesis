from __future__ import division

__author__='Emily Ahn and Sravana Reddy'

""" Date created: 3/17/2016
    Date modified: 3/28/2016
    ****************************
    Transcribed, phone-based accent classification, using separate GMMs for each accent. 
    See http://www.ece.mcgill.ca/~rrose1/papers/reynolds_rose_sap95.pdf for outline of process.
"""

import numpy as np
from sklearn.mixture import GMM
import os
import sys
#from sklearn.metrics import confusion_matrix
import time
from collections import defaultdict

def train_model(speech_array_list, n_components, covar_type): 
    """train GMM with n_components mixtures from data,
    represented as a list of arrays, where each array contains the
    PLPs from a speech file. Returns this model."""
    bigArray = np.vstack(speech_array_list)
    g = GMM(n_components=n_components,covariance_type=covar_type)
    g.fit(bigArray)
    return g

def apply_model(gmm_model, speech_array):
    """compute total log probability (sum across all rows) of
    speech_array under this model"""
    # assume that each time stamp's log-probs are INDEPENDENT
    return np.sum(gmm_model.score(speech_array))

def load_data(phonedir, langlist):
    data = {}
    # for each Lang, create dictionary in "data"                                                     
    for li, lang in enumerate(langlist):
        langstart = time.time()
        data[lang] = {}
        # each lang has a dictionary of phones corresponding to 
        for filename in os.listdir(os.path.join(phonedir, lang)):
            phone = os.path.splitext(filename)[0][8:] # only grab phone
            file_noext = os.path.splitext(filename)[0][:8] # only grab filename
            # if not phone=='sil' and not phone=='': # ignore TextGrid phones 'sil' and ''
            # remove stress distinction for vowels. Ex. 'AA1' -> 'AA'
            if len(phone)==3:
                phone = phone[:2]
            if not phone in data[lang]:
                data[lang][phone] = {}
            #data[lang][phone][file_noext] = np.load(os.path.join(phonedir, lang, filename))
            data[lang][phone][file_noext] = list(np.load(os.path.join(phonedir, lang, filename)))

        print 'Loaded compressed data for', lang, time.time() - langstart
    return data

def get_train_data(data, lang, phone):
    """return list of PLP arrays corresponding to 1 phone for filenames in training for this lang"""
    train_files = open(os.path.join('traintestsplit', lang+'.trainlist')).read().split()
    return [data[lang][phone][file_noext] for file_noext in train_files if file_noext in data[lang][phone]]

def run_test(models, data):
    """apply trained models to each file in test data"""
    num_correct = 0.0
    num_total = 0.0

    # initialize lists to put in predictions & results for confusion matrix 
    predicted_labels = []
    actual_labels = []
    langlist = models.keys()
    for ai, actual_lang in enumerate(langlist):
        lang_correct = 0.0
        lang_total = 0.0
        test_files = open(os.path.join('traintestsplit', actual_lang+'.testlist')).read().split()
        for filename in test_files:
            predictions = defaultdict(int)
            logprobs = defaultdict(dict)   # dict: total log prob of this file under each model                    
            for test_lang in langlist:
                #logprobs[test_lang] = {}
                for phone in data[test_lang]:
                    #check for existence. so far do nothing if phone is not in training or models
                    if phone in models[test_lang] and phone in data[actual_lang]: 
                        if filename in data[actual_lang][phone]: #check (e.g. not all AR phones are in this 1 AR file)
                            
                            logprobs[phone][test_lang] = apply_model(models[test_lang][phone], data[actual_lang][phone][filename])
                            
            for phone in logprobs:
                for test_lang in langlist:
                    if test_lang in logprobs[phone]: # check if this phone exists in this lang's training data
                        predictions[test_lang] += logprobs[phone][test_lang]
            
            predicted_lang = max(predictions.items(), key=lambda x:x[1])[0]
            print 'PREDICIONS FOR', filename, predicted_lang
            #print actual_lang, "LOGPROBS:", logprobs
            
            # insert prediction (of lang index) into predicted list                                     

            if actual_lang == predicted_lang:
                num_correct += 1
                lang_correct += 1
            num_total += 1
            lang_total += 1
        print 'results for', actual_lang, lang_correct*100/lang_total

    print 'Accuracy', num_correct*100/num_total

''' #CONFUSION MATRIX (y_test, y_pred) -> (actual label, predictions)                                   
    cm = confusion_matrix(actual_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # display confusion stats by lang (TODO: visualize with matplotlib)                                 
    print '*'*20
    for ai, actual_lang in enumerate(langlist):
        print actual_lang, 'confusion:'
        for pi, predicted_lang in enumerate(langlist):
            print '{0}: {1:.2f}%'.format(predicted_lang, cm_normalized[ai, pi]*100)
        print '*'*20'''

if __name__=='__main__':
    """load .npy data, split into train-test folds, run training and testing"""
    phonedir = sys.argv[1] #'cslu_fae_corpus/phones' # directory with folders of lang phones
    n_components = int(sys.argv[2])  # number of GMM components
    covar = sys.argv[3]    # covar types: ['spherical', 'diag', 'tied', 'full']
    #langlist = ['AR','BP','CA','CZ','FA','FR','GE','HI','HU','IN','IT','JA','KO','MA','MY','PO','PP','RU','SD','SP','SW','TA','VI']
    #langlist = ['AR', 'CZ', 'IN']
    langlist = ['KO', 'MA']
    #langlist = ['AR', 'CZ', 'FR', 'HI', 'KO', 'IN', 'MA']
    #langlist = ['FR', 'HI', 'KO', 'MA']

    start = time.time()
    print "START"
    
    data = load_data(phonedir, langlist) #format: data[lang][phone] = [ [plp1][plp2][plp3]...]
    #print "DATA AR phones AA2", data['AR']['AA2'].keys()
    models = {}   # store trained models for each lang
    for lang in langlist:
        models[lang] = {}
        print "BEGIN TRAINING", lang
        for phone in data[lang]:
            train_langphone_list = get_train_data(data, lang, phone)
            if train_langphone_list: #only build a model if list is not empty
                models[lang][phone] = train_model(train_langphone_list, n_components, covar)
        print 'Trained model for', lang, time.time() - start
    
    # now test
    run_test(models, data)
