"""Write trained GMM model"""

from gmmClassifier import train_model, load_data, get_train_data
import sys
import os
import numpy as np

if __name__=='__main__':
    npzdir = sys.argv[1]
    modeldir = sys.argv[2]
    n_components = sys.argv[3]
    covar = sys.argv[4]
    lang = sys.argv[5]
    
    data = load_data(npzdir, [lang])
    train_lang_list = get_train_data(data, lang)
    
    model = train_model(train_lang_list, int(n_components), covar)
    print 'Trained', covar, 'model for', lang, 'with', n_components

    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    if not os.path.isdir(os.path.join(modeldir, lang)):
        os.mkdir(os.path.join(modeldir, lang))
    
    model.weights_.dump(os.path.join(modeldir, lang,
                                     covar+'-'+n_components+'-'+lang+'.weights'))
    model.means_.dump(os.path.join(modeldir, lang,
                                   covar+'-'+n_components+'-'+lang+'.means'))
    model.covars_.dump(os.path.join(modeldir, lang,
                                    covar+'-'+n_components+'-'+lang+'.covars'))
    print 'Stored model'
    
