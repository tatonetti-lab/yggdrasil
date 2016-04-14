#!/usr/bin/env python

# This is a simple script to run at the iPython prompt to load
# the data into memory (and to import useful libraries).
# Usage: %run load_helper.py

import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import cross_validation
import sklearn
import gc  # garbage collection routines
import os

print "Loading data from binary numpy files..."
if 'X_cat' not in globals():
    print '  loading X_cat'
    X_cat = np.load('../data/preprocessed/X_categorical.npy')
if 'X_num' not in globals():
    print '  loading X_num'
    X_num = np.load('../data/preprocessed/X_numeric.npy')
if 'X_cat_onehot' not in globals():
    print '  loading X_cat_onehot'
    X_cat_onehot = np.load('../data/preprocessed/X_categorical_onehot.npy')
if 'y' not in globals():
    print '  loading y'
    y = np.load('../data/preprocessed/y.npy')
if 'X' not in globals():
    print '  loading X'
    X = np.concatenate((X_num, X_cat_onehot), axis=1)

if not os.path.exists('../data/preprocessed/ready_for_clf'):
    os.makedirs('../data/preprocessed/ready_for_clf')

print "Removing features containing negative values..."
# remove features with negative values (columns 31 and 32):
X_new = np.delete(X, 31, 1)
del(X)
gc.collect()
X_new = np.delete(X_new, 31, 1)
np.save('../data/preprocessed/ready_for_clf/X_new.npy', X_new)

k = 100
print "Selecting {0} best features based on chi-square score...".format(k)
# select 1000 features with best chi square values
X_new2 = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=k).fit_transform(X_new, y)
del(X_new)
gc.collect()

def cv(classifier, params):
    clf = classifier(**params)
    scores = cross_validation.cross_val_score(clf, X_new2,
                                              y, cv=10, scoring='f1_weighted',
                                              verbose=1)
    print sp.stats.describe(scores)


# ENSEMBLE METHODS #
    
# ensemble.GradientBoostingClassifier
# {'learning_rate': 0.1, 'max_features': 5}
# avg. accuracy: 0.852

# ensemble.RandomForestClassifier
# {'n_estimators': 1000, 'max_depth': None, 'min_samples_split': 1}
# avg. accuracy: 0.907


# LINEAR METHODS #
