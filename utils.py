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
<<<<<<< HEAD
    f1score = f1_score(
        y_test, predict(x_test, w, threshold=threshold, proba=False, poly=False)
    )
    return f1score


# Cross validation function
def cross_validation(x, y, initial_w, method, loss, k_fold, hyperparams):
    seed = 12
    k_fold = k_fold
    k_indices = build_k_indices(y, k_fold, seed)

    losses_te = []
    for hp in hyperparams:
        # we iterate over the hyperparameters to test
        loss_te_temp = []
        for k in range(k_fold):
            # We compute the average accuracy in the k_fold for these hyperparameters
            loss_te = cross_validation_one(
                y, x, initial_w, k_indices, k, hp, method, loss
            )
            loss_te_temp.append(loss_te)
        losses_te.append(np.mean(loss_te_temp))
    # We retrieve the hyperparameters minimizing the loss
    best_hp = hyperparams[np.argmin(losses_te)]
    best_loss = losses_te[np.argmin(losses_te)]

    return best_hp, best_loss


# K nearest neighbors classifying technique
def knn(x_train, y_train, x_test, k):
    y_test = np.ones(x_test.shape[0])
    for i, x in enumerate(x_test):
        # for each line in the test set we find the k nearest neighbors
        distances = []
        for j, x_t in enumerate(x_train):
            # we compute the distances to all elements of the train set
            distances.append((np.linalg.norm(x - x_t), j))
        # we sort the distances to get the k lowest
        distances = sorted(distances)
        count = 0
        # we count the number of neighbors which have 1 in y_train
        for nn in distances[:k]:
            if y_train[nn[1]] == 1:
                count += 1
        # we predict the corresponding class
        if count < k / 2:
            y_test[i] = -1
    return y_test


# Calculate hessian for the newton emthod logistic regression
def calculate_hessian(y, tx, w):
    pred = tx.dot(w)
    print(pred.shape)
    pred = np.diag([sigmoid(p) for p in pred])
    r = np.multiply(pred, (1 - pred))
    return tx.T.dot(r).dot(tx) * (1 / y.shape[0])


# Logistic regression with newton method
def logreg_newton_gd(y, tx, w, max_iters, gamma):
    for n_iters in range(max_iters):
        grad = logreg_grad(y, tx, w)
        hess = calculate_hessian(y, tx, w)
        w = w - gamma * np.linalg.solve(hess, grad)
    return w, logreg_loss(y, tx, w)


import numpy as np
=======
    f1score = f1_score(y_test, predict(x_test, w, threshold=threshold))
    return f1score
>>>>>>> 3cd216102a244422e36c04458553b19040e1e586


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


def ftx(x):
    return np.c_[np.ones((x.shape[0], 1)), x]


<<<<<<< HEAD
def predict(x, w, threshold=None, proba=False, poly=False):
    if poly:
        tx = x
    else:
        tx = ftx(x)
=======
def predict(x, w, threshold=None, proba=False):
    tx = x
>>>>>>> 3cd216102a244422e36c04458553b19040e1e586
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
    tx = ftx(x)
    return y, tx
    return y, tx


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
