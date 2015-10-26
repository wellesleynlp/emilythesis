from __future__ import division

# Mini Classifier: AR, HI, MA (Arabic, Hindi, Mandarin)
# Emily Ahn
# 10.19.2015
# Modified from GMM example with Iris data set, credits:
#     Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
#     License: BSD 3 clause

#import matplotlib.pyplot as plt
#import matplotlib as mpl   # SR: no need to plot
import numpy as np

#from sklearn import datasets #do not need this b/c I don't use iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
#Emily's new import
import os
import sys

#leave this method alone. assume 3 classes // 3 accents // 3 colors
def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
# below: iris data for my reference
'''
iris = datasets.load_iris()
#print iris

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(iris.target, n_folds=4)
# Only take the first fold.
train_index, test_index = next(iter(skf))

X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

#is it randomized? ---> NO
#len(y_train)=111, len(y_test)=39
'''

if __name__=='__main__':  # SR wrap into main function
    datadir = sys.argv[1]    # take corpus directory location on command line
    #get all filenames of files
    # SR: changed it to read npz files (can take averages on the fly)
    langs = {'Arabic': os.path.join(datadir, 'npz', 'AR.npz'),
             'Hindi': os.path.join(datadir, 'npz', 'HI.npz'),
             'Mandarin': os.path.join(datadir, 'npz', 'MA.npz')}

    accents_target_names = langs.keys()
    # fill target with int. 0=Arabic, 1=Hindi, 2=Mandarin
    # fill data with PLP values for each file
    accents_target = []
    accents_data = []

    for li, lang in enumerate(accents_target_names):
        npzdata = np.load(langs[lang])   # load from npz file
        for filename in npzdata:
            time_mean = np.mean(npzdata[filename], axis=0)  # take average
            accents_data.append(time_mean)
            accents_target.append(li)
        
    #make sure arrays are numpy arrays!
    accents_target = np.array(accents_target)
    accents_data = np.array(accents_data)

    skf = StratifiedKFold(accents_target, n_folds=4)

    # Try GMMs using different types of covariances.
    # EA: I hope this works without changing it
    n_classes = len(langs)
    
    classifiers = dict((covar_type, GMM(n_components=n_classes,
                            covariance_type=covar_type, init_params='wc', n_iter=20))
                            for covar_type in ['spherical', 'diag', 'tied', 'full'])
    # EA: init_params: 'w' = weights, 'c' = covars
    n_classifiers = len(classifiers)
        
    # SR: let's iterate over all folds and average for each covariance type
    train_acc_average = {}
    test_acc_average = {}
    
    for (train_index, test_index) in skf:

        # note: NOT RANDOM (that's ok). test = 25% of indices, train = other 75%
        X_train = np.array([accents_data[i] for i in train_index])
        y_train = np.array([accents_target[i] for i in train_index])
        X_test = np.array([accents_data[i] for i in test_index])
        y_test = np.array([accents_target[i] for i in test_index])

        # SR: no need to plot
        #plt.figure(figsize=(3 * n_classifiers / 2, 6))
        #plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05, left=.01, right=.99)

        for index, (name, classifier) in enumerate(classifiers.items()):
            # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.
            classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in xrange(n_classes)])

            # Train the other parameters using the EM algorithm.
            classifier.fit(X_train)

            """SR: removing plotting code
            h = plt.subplot(2, n_classifiers / 2, index + 1)
            make_ellipses(classifier, h)

            for n, color in enumerate('rgb'):
            data = accents_data[accents_target == n]
            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=accents_target_names[n])
            # Plot the test data with crosses
            for n, color in enumerate('rgb'):
            data = X_test[y_test == n]
            plt.plot(data[:, 0], data[:, 1], 'x', color=color)
            """
            
            y_train_pred = classifier.predict(X_train)
            train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
            train_acc_average[name] = train_acc_average.get(name, 0) + train_accuracy

            y_test_pred = classifier.predict(X_test)
            test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
            test_acc_average[name] = test_acc_average.get(name, 0) + test_accuracy

    # now average over the folds
    n_folds = len(classifiers)
    for name in classifiers:
        print 'Train accuracy for', name, 'is', train_acc_average[name]/n_folds
        print 'Test accuracy for', name, 'is', test_acc_average[name]/n_folds
        print
        
