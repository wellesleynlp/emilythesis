__author__='Emily Ahn and Sravana Reddy'

""" Accent classification using separate GMMs for each vowel using formants.
    3-way classification between: Arabic, Czech, Indonesian
    Date created: 12.09.2015
    Date modified: 12.10.2015
    TO-DO: Test by file, create look-up dictionary of file and time.
            Current version only tests by vowel.
"""

import numpy as np
from sklearn.mixture import GMM
from collections import defaultdict

# read 1 csv file. create dict sorted by key=vowel, values = [f1, f2, f3, time]
def read_csv(filename):
       
    vowels = np.loadtxt(filename, dtype=str, delimiter = ',', skiprows=1, usecols=(2,))
    f1 = np.loadtxt(filename, dtype=None, delimiter = ',', skiprows=1, usecols=(7,))
    f2 = np.loadtxt(filename, dtype=None, delimiter = ',', skiprows=1, usecols=(8,))
    time = np.loadtxt(filename, dtype=None, delimiter = ',', skiprows=1, usecols=(17,))
    
    f3str = np.loadtxt(filename, dtype=str, delimiter = ',', skiprows=1, usecols=(9,))
    f3 = []
    for value in f3str:
        if value=='None': # a few values in this column are 'None'
            f3.append(2430.) # insert an average float (~avg across AR/CZ/IN)
        else:
            f3.append(float(value))
    f3 = np.array(f3)
    
    #fill data such that key=vowel, value=list of 3 formants + time started
    data = defaultdict(list)
    for i in range(len(vowels)):
        key = vowels[i]
        data[key].append([f1[i], f2[i], f3[i], time[i]])
        
    for key in data.keys():
        print key, len(data[key])
    return data

# given entire dict of data and a float (in seconds, divide between train & test)
# return a tuple of 2 partial dicts --> (train, test)
def split_data(all_data, time_split):
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    for key in all_data.keys():
        for sample in all_data[key]:
            if sample[3] < time_split:
                train_data[key].append(sample)
            else:
                test_data[key].append(sample)
    return (train_data, test_data)
    
# for 1 vowel only, returns 1 GMM
def train_model(formants_list):
    g = GMM(n_components=3,covariance_type='full')
    g.fit(formants_list)
    return g
    
# for all vowels (entire dict of models) for 1 language
def apply_models(model_dict, speech_sample):
    return 0 #log prob

if __name__=='__main__':
    AR_data = read_csv('AR_formants.csv')
    CZ_data = read_csv('CZ_formants.csv')
    IN_data = read_csv('IN_formants.csv')
    # AR: out of 2413 seconds, split train-test at 75%, or at 1809sec
    # CZ: out of 2316 seconds, split train-test at 75%, or at 1737sec
    # IN: out of 2170 seconds, split train-test at 75%, or at 1630sec
    AR_train_data, AR_test_data = split_data(AR_data, 1809.)
    CZ_train_data, CZ_test_data = split_data(CZ_data, 1737.)
    IN_train_data, IN_test_data = split_data(IN_data, 1630.)
    
    AR_models = {}
    CZ_models = {}
    IN_models = {}
    # TRAIN ALL MODELS. loop through 15 vowels
    for vowel in AR_data.keys():
        AR_models[vowel] = train_model(AR_train_data[vowel])
        CZ_models[vowel] = train_model(CZ_train_data[vowel])
        IN_models[vowel] = train_model(IN_train_data[vowel])
    
    # TEST (for now) individual vowel's samples at a time,
    logprobs = defaultdict(dict)
    predicted_lang = defaultdict(dict) 
    # for now, test only on IN_test_data
    print "FOR ALL IN TEST DATA:"
    for vowel in AR_data.keys():
        logprobs[vowel]['AR'] = apply_models(AR_models[vowel], IN_test_data[vowel])
        logprobs[vowel]['CZ'] = apply_models(CZ_models[vowel], IN_test_data[vowel])
        logprobs[vowel]['IN'] = apply_models(IN_models[vowel], IN_test_data[vowel])
    
        predicted_lang[vowel] = max(logprobs[vowel].items(), key=lambda x:x[1])[0]
        print vowel, "predicted:", predicted_lang[vowel]
    '''Note: when testing on all AR only, or all CZ only, or all IN only,
            every vowel is predicted to be CZ
    '''
    
    
    