import numpy as np
import scipy as sp
from sklearn import feature_selection
from sklearn import cross_validation
import sklearn
import gc  # garbage collection routines

DATA_DIR = '../data/preprocessed/ready_for_clf/'

# load training data, training labels, and test data
X = np.load(DATA_DIR + 'X_new.npy')
X_test = np.load(DATA_DIR + 'X_t_new.npy')
y = np.load('../data/preprocessed/y.npy')

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33)


def select_features(X, X_test, y):
    k = 100
    print "Selecting {0} best features based on chi-square score...".format(k)
    # select 1000 features with best chi square values
    sel = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2,
                                                k=k)
    X_2 = sel.fit_transform(X, y)
    del(X)
    gc.collect()

    # filter X_test by selected features in X
    X_test_2 = X_test[:, sel.get_support()]
    del(X_test)
    gc.collect()
    return (X_2, X_test_2)

    
def cv_single_classifier(classifier, params):
    clf = classifier(**params)
    scores = cross_validation.cross_val_score(clf, X_2,
                                              y, cv=10, scoring='f1_weighted',
                                              verbose=1)
    print sp.stats.describe(scores)


def cv(X, y, X_test, k=10):
    """Run k rounds of feature selection followed by
    training and scoring."""
    


def makeoutput(predictions):
    f = open('../output/out.csv', 'w')
    f.write('Id,Prediction\n')
    index = 1
    for i in np.nditer(predictions):
        f.write('{0},{1}\n'.format(index, i))
        index += 1
    f.close()


clf = 
