import numpy as np
import matplotlib.pyplot as plt
import datetime
from helpers import *
from plots import gradient_descent_visualization
from ipywidgets import IntSlider, interact


def generic_regression(y, tx, initial_w, max_iters, gamma, batch_size, grad, loss):
    w = initial_w
    for n_iter in range(max_iters):
        g = grad(y, tx, w, batch_size)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break

    return w, loss(y, tx, w)


def mse_loss(y, tx, w):
    return (np.linalg.norm(y - np.dot(tx, w), ord=2) ** 2) / (2 * y.shape[0])


def mae_loss(y, tx, w):
    return np.linalg.norm(y - np.dot(tx, w), ord=1) / y.shape[0]


def mse_gradient(y, tx, w, batch_size):
    return -np.dot(np.transpose(tx), y - np.dot(tx, w)) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, -1, mse_gradient, mse_loss
    )


def mse_stoch_gradient(y, tx, w, batch_size):
    stochastic_grad = np.zeros((2,))
    for mini_batch_y, mini_batch_x in batch_iter(y, tx, batch_size):
        tmp = mini_batch_y - np.dot(mini_batch_x, w)
        stochastic_grad += np.array([tmp[0], tmp[0] * mini_batch_x[0][1]])
    return -stochastic_grad / batch_size


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, 1, mse_stoch_gradient, mse_loss
    )


def least_squares(y, tx):
    """implement least squares using normal equations"""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = mse_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression"""
    lambda_p = 2 * len(y) * lambda_
    a = tx.T.dot(tx) + lambda_p * np.identity(np.shape(tx)[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mae_loss(y, tx, w)
    return w, loss


def logreg_loss(y, tx, w):
    yp = 1 / (1 + np.exp(np.dot(tx, w)))
    loss = -np.mean(y * (np.log(yp)) - (1 - y) * np.log(1 - yp))
    return loss


def logreg_grad(y, tx, w, batch_size):
    """Méthode de Newton Raphson"""
    yp = 1 / (1 + np.exp(np.dot(tx, w)))
    g = -np.dot(
        np.dot(np.invert(np.dot(np.transpose(tx), np.dot(w, tx))), tx), (y - yp)
    )
    return g


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, -1, logreg_grad, logreg_loss
    )
