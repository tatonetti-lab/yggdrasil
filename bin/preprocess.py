#!/usr/bin/env python

import numpy as np
import scipy as sp
from sklearn import cross_validation
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
                    default=0,
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
features = [get_dtype(l) for l in feature_file]

print np.where(features[0] == 'numeric')
