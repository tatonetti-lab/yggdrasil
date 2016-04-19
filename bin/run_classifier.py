import numpy as np
import scipy as sp
from sklearn import feature_selection
from sklearn import cross_validation
import sklearn
import gc  # garbage collection routines


###############
# LOAD THINGS #
###############

DATA_DIR = '../data/preprocessed/'

X_num = np.load(DATA_DIR + 'X_num.npy')
X_cat = np.load(DATA_DIR + 'X_cat_onehot.npy')
X_q_num = np.load(DATA_DIR + 'X_q_num.npy')
X_q_cat = np.load(DATA_DIR + 'X_q_cat_onehot.npy')
y = np.load(DATA_DIR + 'y.npy')


###################
# SELECT FEATURES #
###################

# just jam everything together..
X = np.concatenate((X_num, X_cat), axis=1)
X_q = np.concatenate((X_q_num, X_q_cat), axis=1)

############################
# TRAIN and CROSS VALIDATE #
############################

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33)

# def cv_single_classifier(classifier, params):
#     clf = classifier(**params)
#     scores = cross_validation.cross_val_score(clf, X_2,
#                                               y, cv=10, scoring='f1_weighted',
#                                               verbose=1)
#     print sp.stats.describe(scores)


# def cv(X, y, X_test, k=10):
#     """Run k rounds of feature selection followed by
#     training and scoring."""


############################
# PREDICT and WRITE OUTPUT #
############################


def makeoutput(predictions):
    f = open('../output/out.csv', 'w')
    f.write('Id,Prediction\n')
    index = 1
    for i in np.nditer(predictions):
        f.write('{0},{1}\n'.format(index, int(i)))
        index += 1
    f.close()
