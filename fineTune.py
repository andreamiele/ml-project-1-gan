from implementations import *
from preprocessing import *
from utils import *
from helpers import *
from score import *

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

gammas = np.logspace(-4,0,15)
max_iters = np.array([100,250,400])
features = np.arange(40, 60, 5)
degrees = np.array([1,2,3])
thresholds = np.arange(0.5, 0.90, 0.1)
bf1 = 0
bhps = []

for f in features:
  x_t = kbest(x_train, f, fscores)
  y_t = y_train
  k_indices = build_k_indices(y_t, 4, 12)
  for degree in degrees:
    x_tr = build_poly(x_t, degree)
    perm = np.random.permutation(x_train.shape[0])
    separation = int(np.floor(4*perm.shape[0]/5))
    x_verif = x_tr[perm[separation:],:]
    y_verif = y_train[perm[separation:]]
    x_tr = x_tr[perm[:separation],:]
    y_tr = y_train[perm[:separation]]
    w_init = np.ones(x_tr.shape[1])
    for max_iter in max_iters:
      for gamma in gammas:
        for threshold in thresholds:
          print([f,degree,max_iter,gamma, threshold])
          w,_ = mean_squared_error_gd(y_tr, x_tr, w_init, max_iter, gamma)
          f1score = f1_score(y_verif, predict(x_verif, w, threshold=threshold))
          if bf1 < f1score:
            print(f1score)
            bf1 = f1score
            bhps = [f,degree,max_iter,gamma, threshold]
          
          #loss_te_temp = []
          #for k in range(4):
          #  loss_te = cross_validation_one(y_train, x_tr, w_init, k_indices, k, threshold, [max_iter, gamma], mean_squared_error_gd_CV)
          #  loss_te_temp.append(loss_te)
          #losses.append(np.mean(loss_te_temp))
          
print("hyperparameters ([nb of feature, degree of poly, max_iter, gamma, threshold]):", bhps)

#bhps = [50, 1, 100, 0.1, 0.6]
x_train = kbest(x_train, bhps[0], fscores)
x_test = kbest(x_test, bhps[0], fscores)
x_train = build_poly(x_train, bhps[1])
x_test = build_poly(x_test, bhps[1])
w_init = np.ones(x_train.shape[1])
w,_ = mean_squared_error_gd(y_train, x_train, w_init, bhps[2], bhps[3])
y_pred = predict(x_test, w, bhps[4])
create_csv_submission(test_ids, y_pred, "gd_fine_tuned.csv")