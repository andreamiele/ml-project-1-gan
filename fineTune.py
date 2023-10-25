from implementations import *
from preprocessing import *
from utils import *
from helpers import *
from score import *

def ridge_regression_CV(y, tx, w, lbda):
  return ridge_regression(y, tx, lbda)[0]

def build_poly(x, degree):
    # Add a polynomial basis function to the data x, up to a  #
    # degree=degree                                           #
    ret = np.ones([len(x), 1])
    for d in range(1, degree + 1):
        ret = np.c_[ret, np.power(x, d)]

    return ret

def partialPreprocessing(X_train, X_test, Y_train, sampling_strat1=0.105, sampling_strat2=0.5):
    imp = SimpleImputer()
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp = imp.fit(X_test)
    X_test = imp.transform(X_test)

    X_t = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
    X_t2 = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)
    ros = RandomOverSampler(sampling_strategy=sampling_strat1)
    X_balanced, Y_balanced = ros.fit_generate(X_t, Y_train)
    rus = RandomUnderSampler(sampling_strategy=sampling_strat2)
    X_t, Y_t = rus.fit_resample(X_balanced, Y_balanced)
    X_t = standardize(X_t)
    X_t2 = standardize(X_t2)
    return X_t, Y_t, X_t2

def predict_labels_threshold(weights, data, threshold):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    thresholds = np.quantile(y_pred, threshold)
    y_pred[np.where(y_pred <= thresholds)] = -1
    y_pred[np.where(y_pred > thresholds)] = 1
    return y_pred

def kbest(x_train, k, fscores):
  idx = np.argpartition(fscores, k)
  return x_train[:,idx[:k]]

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)
x_train, y_train, x_test = partialPreprocessing(x_train, x_test, y_train)

fscores = np.loadtxt("f_scores_after_strat105_500.csv")

lambdas = np.logspace(-5, 0, 15)
features = np.arange(30, 190, 40)
degrees = np.array([2,3,4])
hps = []
losses = []
for f in features:
  x_t = kbest(x_train, f, fscores)
  y_t = y_train
  k_indices = build_k_indices(y_t, 4, 12)
  for degree in degrees:
    x_tr = build_poly(x_t, degree)
    for lbda in lambdas:
      print([f,degree,lbda])
      hps.append([f,degree,lbda])
      loss_te_temp = []
      for k in range(4):
        loss_te = cross_validation_one(y_train, x_tr, 0, k_indices, k, lbda, ridge_regression_CV)
        loss_te_temp.append(loss_te)
      losses.append(np.mean(loss_te_temp))

bhps = hps[np.argmax(losses)]
print("hyperparameters ([nb of feature, degree of poly, lambda]):", bhps)

x_train = kbest(x_train, bhps[0], fscores)
x_test = kbest(x_test, bhps[0], fscores)
x_train = build_poly(x_train, bhps[1])
x_test = build_poly(x_test, bhps[1])
w,_ = ridge_regression(y_train, x_train, bhps[2])
y_pred = predict_labels_threshold(w, x_test, 0.5)
create_csv_submission(test_ids, y_pred, "ridge_fine_tuned.csv")