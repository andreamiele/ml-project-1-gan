import numpy as np


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
    tx = tx(x)
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
