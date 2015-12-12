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
    g = GMM(n_components=1,covariance_type='full')
    g.fit(formants_list)
    return g
    
# for one vowel for 1 language
def apply_models(model, speech_samples):
    return np.sum(model.score(speech_samples))

if __name__=='__main__':
    lang_list = ['AR', 'CZ', 'IN']
    # AR: out of 2413 seconds, split train-test at 75%, or at 1809sec
    # CZ: out of 2316 seconds, split train-test at 75%, or at 1737sec
    # IN: out of 2170 seconds, split train-test at 75%, or at 1630sec
    time_list = {'AR':1809., 'CZ':1737., 'IN':1630.}
    lang_data = {lang: read_csv(lang+'_formants.csv') for lang in lang_list}
    train_test = {lang: split_data(lang_data[lang], time_list[lang]) for lang in lang_list}
    models = {}
    # TRAIN ALL MODELS. loop through 15 vowels
    for lang in lang_list:
        models[lang] = {}
        
        for vowel in lang_data['AR'].keys():
            models[lang][vowel] = train_model(train_test[lang][0][vowel])
    
    # TEST (for now) individual vowel's samples at a time,
    logprobs = defaultdict(dict)
    predicted_lang = defaultdict(dict) 

    for test_lang in lang_list:
        logprobs = defaultdict(dict)
        predicted_lang = defaultdict(dict) 
        for vowel in lang_data['AR'].keys():
            for train_lang in lang_list:
                logprobs[vowel][train_lang] = apply_models(models[train_lang][vowel], train_test[test_lang][1][vowel])
        
            predicted_lang[vowel] = max(logprobs[vowel].items(), key=lambda x:x[1])[0]
            print "LOGPROB", logprobs[vowel]
            print vowel, "actual:", test_lang, "predicted:", predicted_lang[vowel]
    
    
    