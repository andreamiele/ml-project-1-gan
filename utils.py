import numpy as np
from implementations import *
from score import *

# Build the k_indices for the k_fold
def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


# Function to do one run of cross_validation
def cross_validation_one(y, x, initial_w, k_indices, k, threshold, hp, method):
    # We retrieve the fold we consider as test
    x_test = x[k_indices[k], :]
    y_test = y[k_indices[k]]
    train_indices = np.array([])
    i = 0
    # We take the rest as train set
    for i in range(len(k_indices)):
        if i == k:
            continue
        train_indices = np.append(train_indices, k_indices[i])

    x_train = x[train_indices.astype(int)]
    y_train = y[train_indices.astype(int)]

    # We train a model and return its loss
    w = method(y_train, x_train, initial_w, hp)
    f1score = f1_score(y_test, predict(x_test, w, threshold=threshold))
    return f1score


def standardize(x):
    """Standardize the original data set."""
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x


def build_poly(x, degree):
    # Add a polynomial basis function to the data x, up to a  #
    # degree=degree                                           #
    ret = np.ones([len(x), 1])
    for d in range(1, degree + 1):
        ret = np.c_[ret, np.power(x, d)]

    return ret


def tx(x):
    return np.c_[np.ones((x.shape[0], 1)), x]


def predict(x, w, threshold=None, proba=False):
    tx = x
    if proba:
        predictions = sigmoid(tx.dot(w))
    else:
        predictions = tx.dot(w)
    if threshold == None:
        return [1 if prediction > 0.5 else -1 for prediction in predictions]
    else:
        threshold_ = np.quantile(predictions, threshold)
        return [1 if prediction > threshold_ else -1 for prediction in predictions]


def build_model_data(x, y):
    # Form (y,tX) to get regression data in matrix form.      #
    tx = tx(x)
    return y, tx
