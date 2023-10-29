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
    return X_t, Y_t, X_t2

def kbest(x_train, k, fscores):
  idx = np.argpartition(fscores, k)
  return x_train[:,idx[:k]]


max_iters = [1000,1100,1200]
learn_rate = np.arange(0.015, 0.026, 0.001) # Learning Rates
d_ = [1]  # Max Rates
k_ = np.arange(306,315,1)  # K for Select k best
r_ = [0.001 * i for i in range(600,900)]  # Threshold in quantile
print(r_)

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
fscores = np.loadtxt("f_scores_after_strat105_500.csv")
test_ids = test_ids.astype(dtype=int)
f1_ = 0
params = [0, 0, 0, 0, 0, 0]
"""
x_train, x_test, y_train, y_test = split_data(y_train, x_train, 0.75)

for k in k_:
  x_train2, yb_train2, x_test2 = partialPreprocessing(
    x_train, x_test, y_train
  )
  x_train2 = kbest(x_train2, k, fscores)
  x_test2 = kbest(x_test2, k, fscores)
  x_train2 = standardize(x_train2)
  x_test2 = standardize(x_test2)
  
  for d in d_:
    tx = build_poly(x_train2, degree=d)
    tx_test = build_poly(x_test2, degree=d)
    initial_w = np.ones(tx.shape[1])
    for mi in max_iters:
      for l in learn_rate:
        w, _ = mean_squared_error_gd(yb_train2, tx, initial_w, mi, l)
        for r in r_:
          y_pred = predict(
            tx_test, w, threshold=r, proba=False, poly=True
          )
          f1 = f1_score(y_test, y_pred)
          if f1 > f1_:
            f1_ = f1
            params = [k, d, mi, l, r]
            print(params, f1)
"""

params = [312, 1, 1200, 0.02500000000000001, 0.861] 
f1_ = 0.40205468457381344
  
print(params, f1_)           
x_train2, yb_train2, x_test2 = partialPreprocessing(
  x_train, x_test, y_train
)
k = params[0]
mi = params[2]
gamma = params[3]
r = params[4]
x_train2 = kbest(x_train2, k, fscores)
x_test2 = kbest(x_test2, k, fscores)
x_train2 = standardize(x_train2)
x_test2 = standardize(x_test2)
yb_train2, tx = build_model_data(x_train2, yb_train2)
initial_w = np.zeros(tx.shape[1])
w, loss = mean_squared_error_gd(yb_train2, tx, initial_w, mi, gamma)
y_pred = predict(x_test2, w, threshold=r, proba=False, poly=False)
create_csv_submission(test_ids, y_pred, "gd_fine_tuned.csv")