#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn import cross_validation
from itertools import compress  # for filtering by boolean array
import argparse

parser = argparse.ArgumentParser(
    description='Preprocess data for Kaggle competition.')
parser.add_argument('-i', '--input_file',
                    default="../data/data.csv")
parser.add_argument('-o', '--quiz_file',
                    default="../data/quiz.csv")
parser.add_argument('-f', '--features',
                    default="../data/field_types.txt")
parser.add_argument('-v', '--verbosity',
                    action="count",
                    default=1,
                    help="increase output verbosity")

args = parser.parse_args()

if args.verbosity >= 1:
    print "Reading data from input file: {0}".format(args.input_file)
data = np.genfromtxt(args.input_file, delimiter=',', dtype=np.str_)

# read features file and determine type for each feature column
dt = np.dtype([('numeric', np.float64), ('categorical', np.str_, 16)])


def get_dtype(line):
    """Note: doesn't take number of levels into account"""
    words = line.split()
    if words[1] == 'numeric':
        print 'numeric'
        return ('numeric', None)
    else:
        print 'categorical'
        return ('categorical', len(words) - 1)
feature_file = tuple(open(args.features, 'r'))
print "Types of features present:"
features = [get_dtype(l) for l in feature_file]

# split into two matrices and extract labels
f, _ = zip(*features)
f = np.array(f)
n = (f == 'numeric')
X_numeric = data[:, :-1][:, n][1:, :].astype(float)
X_categorical = data[:, :-1][:, np.invert(n)][1:, :]
y = data[:, (data.shape[1]-1)][1:].astype(float)
np.save('../data/preprocessed/X_numeric.npy', X_numeric)
np.save('../data/preprocessed/X_categorical.npy', X_categorical)
np.save('../data/preprocessed/y.npy', y)

# encode categorical features
le = sklearn.preprocessing.LabelEncoder()
a = [x[0] == 'categorical' for x in features]
to_encode = list(zip(*list(compress(features, a)))[1])
integerized = np.apply_along_axis(le.fit_transform, 0, X_categorical)
enc = sklearn.preprocessing.OneHotEncoder(n_values=to_encode)
enc.fit_transform(integerized)
