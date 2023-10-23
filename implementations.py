import numpy as np
from helpers import *


# Function taking a gradient and a loss functions as argument and implementing a generic regression
def generic_regression(y, tx, initial_w, max_iters, gamma, grad, loss):
    w = initial_w
    for n_iter in range(max_iters):
        g = grad(y, tx, w)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break

    return w, loss(y, tx, w)


def mse_loss(y, tx, w):
    return np.float64(
        (np.linalg.norm(y - np.dot(tx, w), ord=2) ** 2) / (2 * y.shape[0])
    )


def mae_loss(y, tx, w):
    return np.float64(np.linalg.norm(y - np.dot(tx, w), ord=1) / y.shape[0])


def mse_gradient(y, tx, w):
    return -np.dot(np.transpose(tx), y - np.dot(tx, w)) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, mse_gradient, mse_loss
    )


def mse_stoch_gradient(y, tx, w):
    index = np.random.randint(0, y.shape[0])
    tmp = y[index] - tx[index].dot(w)
    return -tmp * tx[index].T


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, mse_stoch_gradient, mse_loss
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
    loss = mse_loss(y, tx, w)
    return w, loss


def positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def negative_sigmoid(x):
    tmp = np.exp(x)
    return tmp / (1 + tmp)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logreg_loss(y, tx, w):
    return np.sum(np.sum(np.logaddexp(0, tx.dot(w)) - y*(tx.dot(w))))


def logreg_grad(y, tx, w):
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, logreg_grad, logreg_loss
    )


def reg_logreg_grad(y, tx, w, lambda_):
    grad = logreg_grad(y, tx, w)
    return grad + lambda_ * w


def reg_logreg_loss(y, tx, w, lambda_):
    loss = logreg_loss(y, tx, w)
    return np.float64(loss + (lambda_ * np.linalg.norm(w) ** 2) / 2)


def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    for n_iter in range(max_iters):
        g = reg_logreg_grad(y, tx, w, lambda_)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break

    return w, logreg_loss(y, tx, w)


"""
def logreg_loss(y, tx, w):
    yp = 1 / (1 + np.exp(np.dot(tx, w)))
    loss = -np.mean(y * (np.log(yp)) - (1 - y) * np.log(1 - yp))
    return loss


def logreg_grad(y, tx, w, batch_size):
    #Méthode de Newton Raphson
    yp = 1 / (1 + np.exp(np.dot(tx, w)))
    g = -np.dot(
        np.dot(np.invert(np.dot(np.transpose(tx), np.dot(w, tx))), tx), (y - yp)
    )
    return g


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, -1, logreg_grad, logreg_loss
    )
"""
