import numpy as np
from helpers import *


# Function taking a gradient and a loss functions as argument and implementing a generic regression
def generic_regression(y, tx, initial_w, max_iters, gamma, grad, loss):
    """Generic regression function for model training

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy array: training dataset
        initial_w (D,) numpy array: initial weight vector
        max_iters (int): maximum number of iterations for the regression
        gamma (float): learning rate
        grad ((y, tx, w) -> (D,) numpy array): gradient function for the given regression
        loss ((y, tx, w) -> float): loss function for the given regression

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    w = initial_w
    for n_iter in range(max_iters):
        g = grad(y, tx, w)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break

    return w, loss(y, tx, w)


def mse_loss(y, tx, w):
    """Mean Squared Error

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        w (D,) numpy array: initial weight vector

    Returns:
        float: the Mean Squared Error for y, tx, w
    """
    return np.float64(
        (np.linalg.norm(y - np.dot(tx, w), ord=2) ** 2) / (2 * y.shape[0])
    )


def mae_loss(y, tx, w):
    """Mean Absolute Error

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        w (D,) numpy array: initial weight vector

    Returns:
        float: the Mean Absolute Error for y, tx, w
    """
    return np.float64(np.linalg.norm(y - np.dot(tx, w), ord=1) / y.shape[0])


def mse_gradient(y, tx, w):
    """Gradient of the MSE error

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        w (D,) numpy array: initial weight vector

    Returns:
        (D,) numpy array: the gradient at y, tx, w of the MSE error
    """
    return -np.dot(np.transpose(tx), y - np.dot(tx, w)) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent with Mean Squared Error

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        initial_w (D,) numpy array: initial weight vector
        max_iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, mse_gradient, mse_loss
    )


def mse_stoch_gradient(y, tx, w):
    """Gradient of the MAE error

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        w (D,) numpy array: initial weight vector

    Returns:
        (D,) numpy array: the gradient at y, tx, w of the MAE error
    """
    index = np.random.randint(0, y.shape[0])
    tmp = y[index] - tx[index].dot(w)
    return -tmp * tx[index].T


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient descent with Mean Squared Error

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        initial_w (D,) numpy array: initial weight vector
        max_iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, mse_stoch_gradient, mse_loss
    )


def least_squares(y, tx):
    """Least squares method

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = mse_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """Ridge regression method

    Args:
        y (N,) numpy array: training set example
        tx (N,D) numpy arrray: training dataset
        lambda_ (float): regularisation factor

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    lambda_p = 2 * y.shape[0] * lambda_
    a = tx.T.dot(tx) + lambda_p * np.identity(np.shape(tx)[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mse_loss(y, tx, w)
    return w, loss


def positive_sigmoid(x):
    """Sigmoid for positive entries

    Args:
        x (float)

    Returns:
        float
    """
    return 1 / (1 + np.exp(-x))


def negative_sigmoid(x):
    """Sigmoid for negative entries

    Args:
        x (float)

    Returns:
        float
    """
    tmp = np.exp(x)
    return tmp / (1 + tmp)


def sigmoid(t):
    """Sigmoid function

    Args:
        t (N,) numpy array: x.dot(w)

    Returns:
        (N,) numpy array: prediction for x,w
    """
    positive = t >= 0
    negative = ~positive
    res = np.empty_like(t, dtype=np.float64)
    res[positive] = positive_sigmoid(t[positive])
    res[negative] = negative_sigmoid(t[negative])
    return res


def logreg_loss(y, tx, w):
    """Loss for the logistic regression

    Args:
        y (N,): train examples
        tx (N,D): train set
        w (D,): weights vector

    Returns:
        float: loss for y, tx, w
    """
    return np.sum(np.sum(np.logaddexp(0, tx.dot(w)) - y * (tx.dot(w)))) / y.shape[0]


def logreg_grad(y, tx, w):
    """Gradient of the logistic loss

    Args:
        y (N,): train examples
        tx (N,D): train set
        w (D,): weights vector

    Returns:
        (D,) numpy array: Gradient of the logistic loss for y, tx, w
    """
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) / y.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression

    Args:
        y (N,): train examples
        tx (N,D): train set
        w (D,): weights vector
        max_iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        (D,) numpy array, float: Last pair (w, loss)
    """
    return generic_regression(
        y, tx, initial_w, max_iters, gamma, logreg_grad, logreg_loss
    )


def reg_logreg_grad(y, tx, w, lambda_):
    """Gradient for the regularised logistic regression

    Args:
        y (N,): train examples
        tx (N,D): train set
        w (D,): weights vector
        lambda_ (float): regularisation term

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    grad = logreg_grad(y, tx, w)
    return grad + lambda_ * w * 2


def reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma):
    """Regularised logistic regression

    Args:
        y (N,): train examples
        tx (N,D): train set
        lambda_ (float): regularisation term
        w (D,): weights vector
        max_iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        (D,) numpy array, float: last pair (w, loss)
    """
    for n_iter in range(max_iters):
        g = reg_logreg_grad(y, tx, w, lambda_)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break

    return w, logreg_loss(y, tx, w)
