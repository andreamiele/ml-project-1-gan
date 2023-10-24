import numpy as np
from helpers import *

import matplotlib.pyplot as plt


############################ The six methods ##############################


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Gradient descent algorithm.                                         #
    # Define parameters to store w and loss                               #
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = 1 / 2 * np.mean(err**2)

        # gradient w by descent update
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)

    # finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Stochastic gradient descent algorithm.                             #
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1

    # iterate max_iters times, where a small batch is picked on each iteration.
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=1):
            grad, err = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = 1 / 2 * np.mean(err**2)

            w = w - gamma * grad

            # store w and loss
            ws.append(w)
            losses.append(loss)

    # finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]

    return w, loss


def least_squares(y, tx):
    # Calculates the least squares solution, and the loss.            #
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))

    return w, loss


def ridge_regression(y, tx, lambda_):
    # Finds the least square solution with the regulation term        #
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Returns the best w found by logistic regression                 #
    ws = [initial_w]
    losses = []
    w = initial_w
    loss = 0

    # Iterates through the number of steps                            #
    for n_iter in range(max_iters):
        loss = np.sum(np.sum(np.logaddexp(0, tx.dot(w)) - y * (tx.dot(w)))) / y.shape[0]
        prediction = sigmoid(tx.dot(w))
        gradient = tx.T.dot(prediction - y) / y.shape[0]

        # Gradient w by descent update
        w = w - (gamma * gradient)
        ws.append(w)
        losses.append(loss)

    # Finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = 0
        prediction = sigmoid(tx.dot(w))
        loss = sum(
            np.logaddexp(0, tx.dot(w))
            - y * (tx.dot(w))
            + (lambda_ / 2) * np.linalg.norm(w) ** 2
        )
        gradient = tx.T.dot(prediction - y) + (lambda_ * np.linalg.norm(w))

        # gradient w by descent update
        w = w - (gamma * gradient)
        # store w and loss
        ws.append(w)
        losses.append(loss)

    # finds best parameters
    min_ind = np.argmin(losses)
    loss = losses[min_ind]
    w = ws[min_ind][:]

    return w, loss


#######################  Data processing  #######################


def standardize(x):
    """Standardize the original data set."""
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x


def remove999(x_train, pred_train, ids_train, x_test, ids_test):
    # Removes the missing data points (-999 in the data set) and #
    # replaces them with the feature mean                        #
    x_train = np.concatenate((ids_train[:, None], x_train), axis=1)
    x_test = np.concatenate((ids_test[:, None], x_test), axis=1)

    # Fix train data
    sig = x_train[pred_train == 1, :]
    back = x_train[pred_train == -1, :]

    for i in range(1, x_train.shape[1]):
        sig_mean = sum(sig[sig[:, i] != -999, i]) / len(sig[sig[:, i] != -999, i])
        back_mean = np.sum(back[back[:, i] != -999, i]) / len(
            back[back[:, i] != -999, i]
        )
        test_mean = np.sum(x_test[x_test[:, i] != -999, i]) / len(
            x_test[x_test[:, i] != -999, i]
        )
        train_mean = np.sum(x_train[x_train[:, i] != -999, i]) / len(
            x_train[x_train[:, i] != -999, i]
        )

        sig[sig[:, i] == -999, i] = train_mean
        back[back[:, i] == -999, i] = train_mean
        x_test[x_test[:, i] == -999, i] = test_mean

    x_train_fixed = np.vstack((sig, back))
    x_train_fixed = x_train_fixed[
        x_train_fixed[:, 0].argsort(),
    ]

    x_train_fixed = x_train_fixed[:, 1:]
    x_test_fixed = x_test[:, 1:]

    return x_train_fixed, x_test_fixed


def removecols(input_data_train, input_data_test, cols):
    # Removes any columns from the train and test data set     #
    input_data_train = np.delete(input_data_train, cols, axis=1)
    input_data_test = np.delete(input_data_test, cols, axis=1)

    return input_data_train, input_data_test


def logpositive(x_train, x_test):
    # Converts features with only positive entries into their    #
    # logarithms                                                #
    for i in range(1, x_train.shape[1]):
        if np.all(x_train[:, i]) > 0 and np.all(x_test[:, i] > 0):
            x_train[:, i] = np.log10(x_train[:, i])
            x_test[:, i] = np.log10(x_test[:, i])

    return x_train, x_test


def findcorrelation(pred_train, sig_or_back):
    col = []
    for i in range(1, pred_train.shape[1]):
        if abs(np.cov(pred_train[:, i], sig_or_back)[0, 1]) < 0.005:
            col.append(i)

    return col


################### Other implementations used ################


def build_model_data(x, y):
    # From ex02 template                                      #
    # Form (y,tX) to get regression data in matrix form.      #
    N = len(y)
    tx = np.c_[np.ones((y.shape[0], 1)), x]

    return y, tx


def compute_mse(y, tx, w):
    # Compute the loss by mse.                                #
    e = y - tx.dot(w)
    mse = 1 / (2 * len(e)) * e.dot(e)

    return mse


def compute_gradient(y, tx, w):
    # Compute the gradient at a certain point w               #
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)

    return grad, err


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=float)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


def build_poly(x, degree):
    # Add a polynomial basis function to the data x, up to a  #
    # degree=degree                                           #
    ret = np.ones([len(x), 1])
    for d in range(1, degree + 1):
        ret = np.c_[ret, np.power(x, d)]

    return ret


def split_data(y, x, ratio, seed=10):
    # Splitting data into train and test set, with (share =   #
    # ratio) of the data in the training set                  #
    np.random.seed(seed)
    N = len(y)

    index = np.random.permutation(N)
    index_tr = index[: int(np.floor(N * ratio))]
    index_te = index[int(np.floor(N * ratio)) :]
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    return x_tr, x_te, y_tr, y_te


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    # From ex02 template

    # Generate a minibatch iterator for a dataset.           #
    # Takes as input two iterables (here the output desired  #
    # values 'y' and the input data 'tx') Outputs an         #
    # iterator which gives mini-batches of `batch_size`      #
    # matching elements from `y` and `tx`. Data can be       #
    # randomly shuffled to avoid ordering in the original    #
    # data messing with the randomness of the minibatches.   #

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
