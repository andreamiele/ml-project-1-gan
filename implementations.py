import numpy as np
from helpers import *

<<<<<<< HEAD
=======

>>>>>>> 865b7da5cec3d63df9e18d00dbe13381f7221902
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
    return np.float64((np.linalg.norm(y - np.dot(tx, w), ord=2) ** 2) / (2 * y.shape[0]))


def mae_loss(y, tx, w):
    return np.float64(np.linalg.norm(y - np.dot(tx, w), ord=1) / y.shape[0])


def mse_gradient(y, tx, w):
    return -np.dot(np.transpose(tx), y - np.dot(tx, w)) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, mse_gradient, mse_loss
    )


def mse_stoch_gradient(y, tx, w):
<<<<<<< HEAD
    index = np.random.randint(0, y.shape[0] - 1)
    tmp = y[index] - tx[index].dot(w)
    return -(tmp**2) / 2
=======
    index = np.random.randint(0, y.shape[0])
    tmp = y[index] - tx[index].dot(w)
    return -tmp * tx[index].T
>>>>>>> 865b7da5cec3d63df9e18d00dbe13381f7221902


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


<<<<<<< HEAD
def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)
=======
def positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def negative_sigmoid(x):
    tmp = np.exp(x)
    return tmp / (1 + tmp)
>>>>>>> 865b7da5cec3d63df9e18d00dbe13381f7221902


def sigmoid(x):
    positive = x >= 0
<<<<<<< HEAD
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=float)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result
=======
    negative = ~positive
    res = np.empty_like(x, dtype=np.float64)
    res[positive] = positive_sigmoid(x[positive])
    res[negative] = negative_sigmoid(x[negative])
    return res
>>>>>>> 865b7da5cec3d63df9e18d00dbe13381f7221902


def logreg_loss(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + 1e-5)) + (1 - y).T.dot(np.log(1 - pred + 1e-5))
    return np.float64(np.squeeze(-loss).item() * (1 / y.shape[0]))



def logreg_grad(y, tx, w):
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) * (1 / y.shape[0])


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, logreg_grad, logreg_loss
    )


def reg_logreg_grad(y, tx, w, lambda_):
    grad = logreg_grad(y, tx, w)
    return grad + lambda_ * w



def reg_logreg_loss(y, tx, w, lambda_):
    loss = logreg_loss(y, tx, w)
<<<<<<< HEAD
    return loss + lambda_ * np.linalg.norm(w) ** 2
=======
    return np.float64(loss + (lambda_ * np.linalg.norm(w) ** 2) / 2)
>>>>>>> 865b7da5cec3d63df9e18d00dbe13381f7221902


def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    for n_iter in range(max_iters):
        g = reg_logreg_grad(y, tx, w, lambda_)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break

<<<<<<< HEAD
    return w, reg_logreg_loss(y, tx, w, lambda_)
=======
    return w, logreg_loss(y, tx, w)
>>>>>>> 865b7da5cec3d63df9e18d00dbe13381f7221902


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
