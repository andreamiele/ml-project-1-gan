from helpers import *
from utils import *
from implementations import *
import numpy as np
import matplotlib.pyplot as plt

y_train = np.genfromtxt("y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0)
x_train = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)
x_test = np.genfromtxt("x_test_processed.csv", delimiter=" ", skip_header=0)
ids = np.genfromtxt("test_ids.csv", delimiter=",")


def logreg_grad_sgd(y, tx, w):
    index = np.random.randint(0,y.shape[0]-1)
    pred = sigmoid(tx[index].dot(w))
    return tx[index]*(pred - y[index])

def logistic_regression_sgd(tx, y, initial_w, hyperparameters):
    max_iters = hyperparameters[0]
    gamma = hyperparameters[1]
    w = initial_w
    for n_iter in range(max_iters):
        g = logreg_grad_sgd(y, tx, w)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break
    return w

def reg_logistic_regression_CV_friendly(y, tx, w, hyperparameters):
    return reg_logistic_regression(y, tx, hyperparameters[0], w, hyperparameters[1], hyperparameters[2])

x_test = x_test.T
for col in x_test:
    avg = 0
    card = 0
    for y in col:
        if not np.isnan(y):
            card += 1
            avg += y
    if card != col.shape[0]:
        col[np.isnan(col)] = avg / card
x_test = x_test.T
tmp = np.ones((x_train.shape[0],x_train.shape[1]+1))
tmp[:,1:] = x_train
x_train = tmp

y_train[y_train == -1] = 0

tmp = np.ones((x_test.shape[0],x_test.shape[1]+1))
tmp[:,1:] = x_test
x_test = tmp
w = np.zeros(x_test.shape[1])

gammas = np.arange(200)*0.005
max_iters = np.arange(1000)*100
hyperparameters = []
for mi in max_iters:
    for g in gammas:
        hyperparameters.append([mi,g])
x_train = x_train.astype(np.float128)
y_train = y_train.astype(np.float128)
x_test = x_test.astype(np.float128)
w = w.astype(np.float128)

best_hyperparameters, _ = cross_validation(x_train, y_train, w, logistic_regression_sgd, logreg_loss, 4, hyperparameters)

gamma = best_hyperparameters[1]
max_iters = best_hyperparameters[0]

w = logistic_regression_sgd(x_train, y_train, w, max_iters, gamma)
print(gamma)

pred = sigmoid(x_test.dot(w))
print(np.mean(pred),np.max(pred),np.min(pred))
mean = np.mean(pred)
pred[pred < mean] = -1
pred[pred >= mean] = 1
count = 0
for a in pred:
    if a == -1:
        count += 1
print(count, pred.shape[0])
create_csv_submission(ids, pred, "submit.csv")
