"""Apply trained models to classify test data"""

from sklearn.mixture import GMM
from gmmClassifier import apply_model, run_test, load_data
import sys
import os
import numpy as np

if __name__=='__main__':
    npzdir = sys.argv[1]
    modeldir = sys.argv[2]
    n_components = sys.argv[3]
    covar = sys.argv[4]
    
    #load models
    models = {}
    langlist = ['MA', 'KO', 'IN', 'AR', 'HI', 'CZ', 'FR']
    for lang in langlist:
        if os.path.exists(os.path.join(modeldir, lang, 
                                       covar+'-'+n_components+'-'+lang+'.weights')):
            models[lang] = GMM(n_components=int(n_components), covariance_type=covar)
            models[lang].weights_ = np.load(os.path.join(modeldir, lang, 
                                                         covar+'-'+n_components+'-'+lang+'.weights'))
            models[lang].means_ = np.load(os.path.join(modeldir, lang,
                                                       covar+'-'+n_components+'-'+lang+'.means'))
            models[lang].covars_ = np.load(os.path.join(modeldir, lang,
                                                        covar+'-'+n_components+'-'+lang+'.covars'))
        else:
            print 'Error: model not trained with these parameters for', lang
    
    #load data
    #langlist = models.keys()
    data = load_data(npzdir, langlist)
    run_test(models, data)
