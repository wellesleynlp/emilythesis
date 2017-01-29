""" Using only unigram vectors from transcriptions of accented speech, 
    try to cluster and do accent ID. Clustering includes: 
        - knn (method from knncluster.py in A5 from NLP class)
        - logistic regression (module from sklearn)
    --------------------
    Date created: 01/22/17
    Date modified: 01/29/17
"""

import scipy
import argparse
from scipy.spatial.distance import cdist
from scipy.stats import mode
import codecs
import time
# added
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.csr import csr_matrix
import sys
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy as np

__author__='Emily Ahn'

def knn(trainpoints, traincats, testpoints, k):
    """Given training data points
    and a 1-d array of the corresponding categories of the points,
    predict category for each test point,
    using k nearest neighbors (with cosine distance).
    Return a 1-d array of predicted categories.
    """
    predicted_cat = []
    for point in testpoints:
        dist = scipy.argsort(cdist(trainpoints,[point],'cosine').flatten())
        closest = dist[:k]
        close_cat = scipy.zeros(k, dtype=int)
        for closest_i,i in enumerate(closest) :
            close_cat[closest_i] = traincats[i]
        #close_cat = [traincats[i] for i in closest]
        cat = mode(close_cat)[0]
        predicted_cat.append(cat)

    return predicted_cat

    
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

if __name__=='__main__':
    
    thesis_dir = '..'#'Desktop/thesis/emilythesis'
    with open(thesis_dir+'/traintestsplit/7lang.trainlist.sorted.20sec', 'r') as reader:
        trainlist = [line.replace('\n','') for line in reader.readlines()]
    with open(thesis_dir+'/traintestsplit/7lang.testlist.sorted.20sec', 'r') as reader:
        testlist = [line.replace('\n','') for line in reader.readlines()]
        
    langlist = ['AR','CZ','FR','HI','IN','KO','MA']
                
    train_filelist, traincats = get_flist_labels(trainlist, '..')
    test_filelist, testcats = get_flist_labels(testlist, '..')
    
    all_filelist = train_filelist
    all_filelist.extend(test_filelist)

    
    cv = CountVectorizer(input='filename') #optional param: stop_words='english'
    td_matrix = cv.fit_transform(all_filelist).toarray()
    # dimensions = 1226 (# train+test files) x 4182 (dimensions)
    # without toarray, it's <class 'scipy.sparse.csr.csr_matrix'>
    #print len(cv.get_feature_names())
    
    trainpoints = td_matrix[:len(traincats)]
    testpoints = td_matrix[-len(testcats):]
    
    # run knn classifier
    #k = int(sys.argv[1])
    #predictions = knn(trainpoints, traincats, testpoints, k)
    
    # run logistic regresion clasifier
    logreg = linear_model.LogisticRegression()
    print 'open logreg'
    logreg.fit_transform(trainpoints, traincats)
    print 'fitted logreg'
    predictions = logreg.predict(testpoints)
    print 'predicted logreg'
    
    # write actual category, predict category, and text of test points, and compute accuracy
    #o = codecs.open('knn.'+str(k)+'.predictions', 'w', 'utf8')4
    o = codecs.open('logreg.predictions', 'w', 'utf8')
    o.write('ACTUAL,PREDICTED,CORRECT?,TEXT\n')
    o.write('{AR:0,CZ:1,FR:2,HI:3,IN:4,KO:5,MA:6}\n')
 
    numcorrect = 0.
    for i, testcat in enumerate(testcats):

        o.write(str(testcats[i]))
        o.write(',')
        o.write(str(predictions[i]))
        o.write(',')
        if testcats[i] == predictions[i]:
            numcorrect += 1
            o.write('CORRECT,')
        else:
            o.write('WRONG,')
        o.write(test_filelist[i]+'\n')

    #print 'Stored predictions in knn.'+str(k)+'.predictions', 'for test points'
    print 'Stored predictions in logreg.predictions', 'for test points'
    acc = numcorrect*100/len(testcats)
    print 'Accuracy: {0:.2f}%'.format(acc)
    
    # CONFUSION MATRIX
    cm = confusion_matrix(testcats, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # display confusion stats by lang (TODO: visualize with matplotlib)                                 
    print '*'*20
    for ai, actual_lang in enumerate(langlist):
        print actual_lang, 'confusion:'
        for pi, predicted_lang in enumerate(langlist):
            print '{0}: {1:.2f}%'.format(predicted_lang, cm_normalized[ai, pi]*100)
        print '*'*20

    
    
    
