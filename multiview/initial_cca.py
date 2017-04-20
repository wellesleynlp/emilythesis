""" Use CCA
    - script modified from baseline_wordvec_cluster.py
    --------------------
    Date created: 04/13/17
    Date modified: 04/19/17
"""

import sys, os
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_decomposition import CCA
#import cPickle

__author__='Emily Ahn'
   
def get_flist_labels(filelist, thesis_dir):
    flist = []
    labels = []
    for item in filelist:
        # filelist = list of filenames containing text info
        # e.g. item='FAR00173'
        lang = item[1:3]
        flist.append(thesis_dir+'/alignments/'+lang+'/'+item+'.lab')
        labels.append(get_langid(lang))
    return (flist, labels)
    
def get_langid(lang):
    ids = {'AR':0,'CZ':1,'FR':2,'HI':3,'IN':4,'KO':5,'MA':6}
    return ids[lang]

def load_data(npzdir, langlist):
    data = {}
    # for each Lang, create dictionary in "data"                                                     
    for li, lang in enumerate(langlist):
        with np.load(os.path.join(npzdir, lang+'.npz')) as x:
            data[lang] = dict(x)
        print 'Loaded compressed data for', lang
    return data

if __name__=='__main__':

    start = time.time()
    print "START"
    
    thesis_dir = '..'#'Desktop/thesis/emilythesis'
    with open(thesis_dir+'/traintestsplit/7lang.trainlist.sorted.20sec', 'r') as reader:
        trainlist = [line.replace('\n','') for line in reader.readlines()]
    with open(thesis_dir+'/traintestsplit/7lang.testlist.sorted.20sec', 'r') as reader:
        testlist = [line.replace('\n','') for line in reader.readlines()]    
    langlist = ['AR','CZ','FR','HI','IN','KO','MA']
    
    ''' LOAD TEXT train/test '''                
    train_flist_labs, traincats = get_flist_labels(trainlist, '..')
    test_flist_labs, testcats = get_flist_labels(testlist, '..')
    
    all_flist_labs = train_flist_labs
    all_flist_labs.extend(test_flist_labs)

    #cv = CountVectorizer(input='filename') #optional param: stop_words='english'
    cv = CountVectorizer(input='filename',stop_words='english')
    td_matrix = cv.fit_transform(all_flist_labs).toarray()
    # dimensions = 1226 (# train+test files) x 4182 (dimensions)
    
    trainpoints_text = td_matrix[:len(traincats)]
    testpoints_text = td_matrix[-len(testcats):]

    print 'LOADED TEXT', time.time() - start

    ''' LOAD SPEECH FT train/text. 
    Concat/flatten 2D array (2046 windows * 52 dim) of each file '''
    trainpoints_sp = []
    testpoints_sp = []

    data = load_data('/home/sravana/data/cslu_fae_corpus/npz',langlist)

    for trainfile in trainlist:
        lang = trainfile[1:3]
        trainpoints_sp.append(data[lang][trainfile+'.npytxt'].flatten())
    # print 'TRAIN_SP SHAPE', np.array(trainpoints_sp).shape

    for testfile in testlist:
        lang = testfile[1:3]
        testpoints_sp.append(data[lang][testfile+'.npytxt'].flatten())
    # print 'TEST_SP SHAPE', np.array(testpoints_sp).shape    

    print 'LOADED SP', time.time() - start
    
    pkl_path = '/home/sravana/data/cslu_fae_corpus/pkl/cca_0417.'
    def save_cca():
        cca = CCA(n_components=2)
        #new = cca.fit_transform(trainpoints_sp,trainpoints_text)
        cca.fit(trainpoints_sp,trainpoints_text)
        
        cca.x_weights_.dump(pkl_path+'x_weights')
        cca.y_weights_.dump(pkl_path+'y_weights')
        cca.x_loadings_.dump(pkl_path+'x_loadings')
        cca.y_loadings_.dump(pkl_path+'y_loadings')
        cca.x_scores_.dump(pkl_path+'x_scores')
        cca.y_scores_.dump(pkl_path+'y_scores')
        cca.x_rotations_.dump(pkl_path+'x_rotations')
        cca.y_rotations_.dump(pkl_path+'y_rotations')
    
    #save_cca()
    #print 'CCA FIT_TRANSFORM & SAVED', time.time() - start
    
    cca = CCA(n_components=2)
    #cca.x_weights_ = np.load(pkl_path+'x_weights')
    #cca.y_weights_ = np.load(pkl_path+'y_weights')
    #cca.x_loadings_ = np.load(pkl_path+'x_loadings')
    #cca.y_loadings_ = np.load(pkl_path+'y_loadings')
    #cca.x_scores_ = np.load(pkl_path+'x_scores')
    #cca.y_scores_ = np.load(pkl_path+'y_scores')
    #cca.x_rotations_ = np.load(pkl_path+'x_rotations')
    #cca.y_rotations_ = np.load(pkl_path+'y_rotations')
    cca.fit_transform(trainpoints_sp,trainpoints_text)
    print 'CCA LOADED', time.time() - start
    
    print 'SCORE', cca.score(testpoints_sp, testcats)
    # ERROR WITH SCORE:
    # sklearn.utils.validation.NotFittedError: This CCA instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.

    print 'DONE', time.time() - start