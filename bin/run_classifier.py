import numpy as np
from sklearn import feature_selection
from sklearn import cross_validation
import sklearn
from sklearn.ensemble import RandomForestClassifier


###############
# LOAD THINGS #
###############

DATA_DIR = '../data/preprocessed/'

print "Loading data"
X_num = np.load(DATA_DIR + 'X_num.npy')
X_cat = np.load(DATA_DIR + 'X_cat_onehot.npy')
X_q_num = np.load(DATA_DIR + 'X_q_num.npy')
X_q_cat = np.load(DATA_DIR + 'X_q_cat_onehot.npy')
y = np.load(DATA_DIR + 'y.npy')


###################
# SELECT FEATURES #
###################

print "Selecting features and creating matrices"
# just jam everything together..
X = np.concatenate((X_num, X_cat), axis=1)
X_q = np.concatenate((X_q_num, X_q_cat), axis=1)

#remove a feature column:
def remcol(n):
    global X
    global X_q
    X = np.delete(X, n, 1)
    X_q = np.delete(X_q, n, 1)

############################
# TRAIN and CROSS VALIDATE #
############################

print "Building cross validation sets"
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33)

print "Training classifier"
clf = RandomForestClassifier(n_estimators=100, verbose=True, n_jobs=4)
# BEST SCORE:
# clf = RandomForestClassifier(n_estimators=200, verbose=True)
clf.fit(X_train, y_train)
print "Performing cross validation"
print "CROSS VALIDATION SCORE: {0}".format(clf.score(X_test, y_test))


############################
# PREDICT and WRITE OUTPUT #
############################

print "Writing output"
def makeoutput(predictions):
    f = open('../output/out.csv', 'w')
    f.write('Id,Prediction\n')
    index = 1
    for i in np.nditer(predictions):
        f.write('{0},{1}\n'.format(index, int(i)))
        index += 1
    f.close()


makeoutput(clf.predict(X_q))

# Get a sorted listing of the most important features in the random forest classifier
s = sorted(zip(map(lambda x: round(x, 10), clf.feature_importances_), range(X_train.shape[1])), reverse=True)
# Use to get the 150 least important features:
_, fs = zip(*s[-50:])
fs = list(fs)
for f in fs:
    print "removing column {0}".format(f)
    remcol(f)
# It'll take a while to remove all of these columns... (VERY long time...)
# Now, retrain a classifier with the trimmed-up data:
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=50, verbose=True, n_jobs=4)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
