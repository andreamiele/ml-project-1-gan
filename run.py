from helpers import *
from implementations import *

y = np.genfromtxt("dataset/y_train.csv", delimiter=",", skip_header=1, usecols=0)
x = np.genfromtxt("dataset/x_train.csv", delimiter=",", skip_header=1)
ids = x[:, 0].astype(np.int64)
x = x[:, 1:]
x[np.where(np.isnan(x))] = 0
y[np.where(y == 0)] = -1

#subsample
x = x[::50]
y = y[::50]
ids = ids[::50]

c = 0
for a in x:
  for b in a:
    if np.isnan(b): c+=1
for b in y:
  if np.isnan(b): c += 1
print(c)

gamma = 0.1
initial_w = np.zeros(x.shape[1])
max_iters = 500
print(x.shape)
print(y.shape)
print(initial_w.shape)

losses, ws = mean_squared_error_gd(y, x, initial_w, max_iters, gamma)
print(ws[:10])
create_csv_submission(ids, np.dot(x, ws[-1]), "submission.csv")