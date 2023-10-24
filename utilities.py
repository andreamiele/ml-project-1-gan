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


def predict_labels_threshold(weights, data, threshold):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    thresholds = np.quantile(y_pred, threshold)
    y_pred[np.where(y_pred <= thresholds)] = -1
    y_pred[np.where(y_pred > thresholds)] = 1

    return y_pred
