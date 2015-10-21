# Mini Classifier: AR, HI, MA (Arabic, Hindi, Mandarin)
# Emily Ahn
# 10.19.2015
# Modified from GMM example with Iris data set, credits:
#     Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
#     License: BSD 3 clause

#print(__doc__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
#Emily's
from os import listdir

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
dirNames = ['cslu_fae_corpus/npytxt_avg/AR', 'cslu_fae_corpus/npytxt_avg/HI', 'cslu_fae_corpus/npytxt_avg/MA']
files_AR = listdir(dirNames[0])
files_HI= listdir(dirNames[1])
files_MA = listdir(dirNames[2])
all_files = files_AR + files_HI + files_MA

# convert allFiles (holds list of fileNames) --> accents_data

# fill target with int. 0=Arabic, 1=Hindi, 2=Mandarin
# fill data with PLP values for each file
accents_target = []
accents_data = []
for i in range(len(all_files)):
    if i < len(files_AR):
        accents_target.append(0)
        with open(dirNames[0]+"/"+all_files[0]) as f:
            line = f.readline()
            one_row = line.split()
            one_row = [float(num) for num in one_row]
            accents_data.append(one_row)
    elif (i < (len(files_AR) + len(files_HI))):
        accents_target.append(1)
        with open(dirNames[1]+"/"+all_files[1]) as f:
            line = f.readline()
            one_row = line.split()
            one_row = [float(num) for num in one_row]
            accents_data.append(one_row)
    else:
        accents_target.append(2)
        with open(dirNames[2]+"/"+all_files[2]) as f:
            line = f.readline()
            one_row = line.split()
            one_row = [float(num) for num in one_row]
            accents_data.append(one_row)
#print "accent data arabic: ", accents_data

skf = StratifiedKFold(accents_target, n_folds=4)
# Only take the first fold.
train_index, test_index = next(iter(skf))

X_train = accents_data[train_index]
y_train = accents_target[train_index]
X_test = accents_data[test_index]
y_test = accents_target[test_index]


'''# IRIS CODE BELOW ***************
n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

plt.figure(figsize=(3 * n_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)

    h = plt.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    for n, color in enumerate('rgb'):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                    label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate('rgb'):
        data = X_test[y_test == n]
        plt.plot(data[:, 0], data[:, 1], 'x', color=color)

    y_train_pred = classifier.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(loc='lower right', prop=dict(size=12))


plt.show()'''