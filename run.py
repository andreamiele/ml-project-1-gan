from helpers import *
from implementations import *

y = np.genfromtxt("dataset/y_train.csv", delimiter=",", skip_header=1, usecols=0)
x = np.genfromtxt("dataset/x_train.csv", delimiter=",", skip_header=1)
x_test = np.genfromtxt("dataset/x_test.csv", delimiter=",", skip_header=1)
x_test = x_test[:, 1:]
x = x[:, 1:]

x_test = x_test[:, np.all(np.logical_not(np.isnan(x)), axis=0)]
x = x[:, np.all(np.logical_not(np.isnan(x)), axis=0)]
print(x.shape)

gamma = 0.1
initial_w = np.zeros(x.shape[1])
max_iters = 500

w = np.linalg.lstsq(x.T.dot(x), x.T.dot(y))[0]
print(w)
np.savetxt("submission.csv", np.dot(x_test,w), header="_MICHD",)