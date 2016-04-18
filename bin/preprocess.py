#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn import cross_validation
from itertools import compress  # for filtering by boolean array
import argparse


# get/bind data
data = np.genfromtxt("../data/data.csv", delimiter=',', dtype=np.str_)
quiz = np.genfromtxt("../data/quiz.csv", delimiter=',', dtype=np.str_)
y = data[1:, -1].astype(float)
data = data[1:, :-1]
quiz = quiz[1:, :]
bind = np.concatenate((data, quiz), axis=0)

feature_file = tuple(open("../data/field_types.txt", 'r'))
features = [get_dtype(l) for l in feature_file]
f, _ = zip(*features)
f = np.array(f)
n = (f == 'numeric')

X_numeric = bind[:, n][1:, :].astype(float)
X_categorical = bind[:, np.invert(n)][1:, :]
#np.save('../data/preprocessed/X{0}_numeric.npy'.format(outprefix), X_numeric)
#np.save('../data/preprocessed/X{0}_categorical.npy'.format(outprefix), X_categorical)

# encode categorical features
le = sklearn.preprocessing.LabelEncoder()
a = [x[0] == 'categorical' for x in features]
to_encode = list(zip(*list(compress(features, a)))[1])
integerized = np.apply_along_axis(le.fit_transform, 0, X_categorical)
enc = sklearn.preprocessing.OneHotEncoder(n_values=to_encode)
X_onehot = enc.fit_transform(integerized).todense()

X_cat_onehot = X_onehot[:126837, :]
X_q_cat_onehot = X_onehot[126837:, :]
X_num = X_numeric[:126836, :]
X_q_num = X_numeric[126836:, :]

np.save('../data/preprocessed/X_num.npy', X_num)
np.save('../data/preprocessed/X_q_num.npy', X_q_num)
np.save('../data/preprocessed/X_cat_onehot.npy', X_cat_onehot)
np.save('../data/preprocessed/X_q_cat_onehot.npy', X_q_cat_onehot)
