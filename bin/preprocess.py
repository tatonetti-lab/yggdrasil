#!/usr/bin/env python

import numpy as np
import sklearn
import sklearn.preprocessing
from itertools import compress  # for filtering by boolean array


def get_dtype(line):
    """Note: doesn't take number of levels into account"""
    words = line.split()
    if words[1] == 'numeric':
        return ('numeric', None)
    else:
        return ('categorical', len(words) - 1)


# get/bind data
data = np.genfromtxt("../data/data.csv", delimiter=',', dtype=np.str_)
quiz = np.genfromtxt("../data/quiz.csv", delimiter=',', dtype=np.str_)
y = data[1:, -1].astype(float)
data_new = data[1:, :-1]
quiz_new = quiz[1:, :]
bind = np.concatenate((data_new, quiz_new), axis=0)

feature_file = tuple(open("../data/field_types.txt", 'r'))
features = [get_dtype(l) for l in feature_file]
f, _ = zip(*features)
f = np.array(f)
n = (f == 'numeric')

X_numeric = bind[:, n].astype(float)
X_categorical = bind[:, np.invert(n)]

# encode categorical features
le = sklearn.preprocessing.LabelEncoder()
a = [x[0] == 'categorical' for x in features]
to_encode = list(zip(*list(compress(features, a)))[1])
integerized = np.apply_along_axis(le.fit_transform, 0, X_categorical)
enc = sklearn.preprocessing.OneHotEncoder(n_values=to_encode)
X_onehot = enc.fit_transform(integerized).todense()

# Split into 4 matrices
X_cat_onehot = X_onehot[:126837, :]
X_q_cat_onehot = X_onehot[126837:, :]
X_num = X_numeric[:126837, :]
X_q_num = X_numeric[126837:, :]

# save everything
np.save('../data/preprocessed/X_num.npy', X_num)
np.save('../data/preprocessed/X_q_num.npy', X_q_num)
np.save('../data/preprocessed/X_cat_onehot.npy', X_cat_onehot)
np.save('../data/preprocessed/X_q_cat_onehot.npy', X_q_cat_onehot)
np.save('../data/preprocessed/y.npy', y)
