from implementations import *
from preprocessing import *
from utils import *
from helpers import *
from score import *

def mean_squared_error_gd_CV(y, tx, w, hps):
  return mean_squared_error_gd(y, tx, w, hps[0], hps[1])[0]

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

def kbest(x_train, k, fscores):
  idx = np.argpartition(fscores, k)
  return x_train[:,idx[:k]]

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)
x_train, y_train, x_test = partialPreprocessing(x_train, x_test, y_train)



fscores = np.loadtxt("f_scores_after_strat105_500.csv")

gammas = np.logspace(-10, -3, 20)
max_iters = np.array([100,250,500,750,1000])
features = np.arange(20, 160, 10)
degrees = np.array([2,3,4])
thresholds = np.arange(0.1, 1, 0.1)
hps = []
losses = []

for f in features:
  x_t = kbest(x_train, f, fscores)
  y_t = y_train
  k_indices = build_k_indices(y_t, 4, 12)
  for degree in degrees:
    x_tr = build_poly(x_t, degree)
    w_init = np.ones(x_train.shape[1])
    for max_iter in max_iters:
      for gamma in gammas:
        for threshold in thresholds:
          print([f,degree,max_iter,gamma])
          hps.append([f,degree,max_iter,gamma])
          loss_te_temp = []
          for k in range(4):
            loss_te = cross_validation_one(y_train, x_tr, 0, k_indices, k, threshold, [max_iter, gamma], mean_squared_error_gd_CV)
            loss_te_temp.append(loss_te)
          losses.append(np.mean(loss_te_temp))

bhps = hps[np.argmax(losses)]
print("hyperparameters ([nb of feature, degree of poly, max_iter, gamma]):", bhps)

x_train = kbest(x_train, bhps[0], fscores)
x_test = kbest(x_test, bhps[0], fscores)
x_train = build_poly(x_train, bhps[1])
x_test = build_poly(x_test, bhps[1])
w_init = np.ones(x_train.shape[1])
w,_ = mean_squared_error_gd(y_train, x_train, w_init, bhps[3], bhps[4])
y_pred = predict(w, x_test, 0.5)
create_csv_submission(test_ids, y_pred, "ridge_fine_tuned.csv")