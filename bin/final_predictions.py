#!/usr/bin/env python

import numpy as np
import sklearn
import sys
import sklearn.preprocessing
from itertools import compress
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


try:
    DATAFILE = sys.argv[1]
    QUIZFILE = sys.argv[2]
    OUTPUTFILE = sys.argv[3]
except:
    print "USAGE: python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE"
    exit(1)


def get_dtype(line):
    """Return tuple of following format:
    ([feature type], [number of features])"""
    words = line.split()
    if words[1] == 'numeric':
        return ('numeric', None)
    else:
        return ('categorical', len(words) - 1)


def makeoutput(predictions):
    """Given a numpy array of predictions, write a file
    in the correct format for the Kaggle competition"""
    f = open(OUTPUTFILE, 'w')
    f.write('Id,Prediction\n')
    index = 1
    for i in np.nditer(predictions):
        f.write('{0},{1}\n'.format(index, int(i)))
        index += 1
    f.close()


# Mask of features that are categorical
n = np.array([False,  True, False, False, False, False,  True, False, False,
              False, False, False, False, False, False,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True, False,
              False, False,  True,  True,  True,  True,  True], dtype=bool)


# Complete listing of all features, with number of distinct values, if 'categorical'
features = [('categorical', 13),
            ('numeric', None),
            ('categorical', 112),
            ('categorical', 2),
            ('categorical', 13),
            ('categorical', 13),
            ('numeric', None),
            ('categorical', 112),
            ('categorical', 2),
            ('categorical', 13),
            ('categorical', 145),
            ('categorical', 4),
            ('categorical', 3031),
            ('categorical', 4),
            ('categorical', 138),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('categorical', 102),
            ('categorical', 102),
            ('categorical', 2090),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None),
            ('numeric', None)]


# LOAD DATA FROM FILES IN ARGV
print "Loading provided data and quiz files..."
data = np.genfromtxt(DATAFILE, delimiter=',', dtype=np.str_)
quiz = np.genfromtxt(QUIZFILE, delimiter=',', dtype=np.str_)
y = data[1:, -1].astype(float)
data_new = data[1:, :-1]
quiz_new = quiz[1:, :]
bind = np.concatenate((data_new, quiz_new), axis=0)


## USED TO CREATE FEATURE DATA STRUCTURES (beware the file path):
# feature_file = tuple(open("../data/field_types.txt", 'r'))
# features = [get_dtype(l) for l in feature_file]
# f, _ = zip(*features)
# f = np.array(f)
# n = (f == 'numeric')


# BIND TRAIN AND QUIZ SETS TO YIELD CONSISTENT FEATURE ENCODING
X_numeric = bind[:, n].astype(float)
X_categorical = bind[:, np.invert(n)]


# ENCODE CATEGORICAL FEATURES: ONE-HOT ENCODING
le = sklearn.preprocessing.LabelEncoder()
a = [x[0] == 'categorical' for x in features]
to_encode = list(zip(*list(compress(features, a)))[1])
integerized = np.apply_along_axis(le.fit_transform, 0, X_categorical)
enc = sklearn.preprocessing.OneHotEncoder(n_values=to_encode)
X_onehot = enc.fit_transform(integerized).todense()


# SPLIT BACK INTO TRAIN AND QUIZ SETS, SEPARATE NUMERIC FROM CATEGORICAL VARS
X_cat = X_onehot[:126837, :]
X_q_cat = X_onehot[126837:, :]
X_num = X_numeric[:126837, :]
X_q_num = X_numeric[126837:, :]


# JAM NUMERICAL AND CATEGORICAL FEATURES TOGETHER FOR RANDOM FOREST
print "Selecting features and creating matrices"
X = np.concatenate((X_num, X_cat), axis=1)
X_q = np.concatenate((X_q_num, X_q_cat), axis=1)


# TRAIN RANDOM FOREST MODEL and CROSS VALIDATE
print "Building cross validation sets"
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33)
print "Training classifier"
print "\n!!!WARNING!!! - this takes quite a long time, \
even on a very fast computer!"
clf = RandomForestClassifier(n_estimators=600, verbose=True,
                             n_jobs=4)
clf.fit(X_train, y_train)
print "Performing cross validation...\n"
print "CROSS VALIDATION SCORE: {0}".format(clf.score(X_test, y_test))


############################
# PREDICT and WRITE OUTPUT #
############################

print "\nPredicting labels on quiz set and priting output to file."

makeoutput(clf.predict(X_q))
