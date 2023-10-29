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
    """Prepare the feature matrix x for use in linear regression model."""
    return np.c_[np.ones((x.shape[0], 1)), x]


def predict(x, w, threshold=None, proba=False, poly=False):
    """Predict, depending if we are using a polynomial expansion or not and if we need a sigmoid or not."""
    if poly:
        tx = x
    else:
        tx = ftx(x)
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


class SimpleImputer:
    def __init__(self):
        """
        Initialize the SimpleImputer.
        """
        self.mean_values = None

    def fit(self, X):
        """
        Compute the mean values for each column in the input data.

        Parameters:
        X : numpy array or array-like
            Input data of shape (n_samples, n_features) where NaN values
            need to be imputed.

        Returns:
        self : SimpleImputer
            The fitted imputer object.
        """
        self.mean_values = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        """
        Replace NaN values in the input data with the mean values computed during fitting.

        Parameters:
        X : numpy array or array-like
            Input data of shape (n_samples, n_features) where NaN values
            need to be imputed.

        Returns:
        X_imputed : numpy array
            Input data with NaN values replaced by the mean values.
        """
        if self.mean_values is None:
            raise ValueError("SimpleImputer has not been fitted. Call fit() first.")

        X_imputed = np.where(np.isnan(X), self.mean_values, X)
        return X_imputed


def local_search_rlr(
    best_hyperparam, fscore, xtrain, ytrain, xtest, ytest, max_jumps, nb_points
):
    """
    Local search for lambda and gamma tuning.
    The idea is to search randomly around the position of grid search optimum.
    At each iteration (maximum number of iterations : max_jumps) :
        - Create a number of new points : for each point in points, add to points [lambda +-0  pas_lambda, gamma +-0 pas_gamma]
        - Here, pas_lambda = saut[0]/ratio et pas_gamma = saut[1]/ratio
        - Calculate f_score for each of these new points and old points (less efficient but clearer)
        - If the best fscore is the same as the old one, multiply ratio by 2
        - Keep in points only nb_points (those with best f-scores)
        - Number of trainings : nb_points*max_jumps*9 (ici le 9 vient du fait que l'on optimise 2 paramètres : 3² = 9)

    Args :
        best_hyperparam (list,np.array) : list of the best hyperparameters found with another method (grid search) that will be used here as a starting point
        xtrain (list,np.array) : preprocessed array of x for training
        ytrain (list,np.array) : corresponding y for training
        xtest (list,np.array) : preprocessed array of x for testing
        ytest (list,np.array) : corresponding y fot testing
        fscore (float) : fscore obtained with hyperparameters : best_hyperparam
        max_jumps (int) : number of iteration
        nb_points (int) : max number of points kept after each iteration

    Out :
        List of best hyperparameters
    """
    w = np.zeros(np.shape(xtrain)[1])

    lambda_best, gamma_best, max_iter = best_hyperparam
    points = [[lambda_best, gamma_best]]
    saut = [lambda_best / 2, gamma_best / 2]
    ratio = 1

    for iter in range(max_jumps):
        print(f"Local search, iteration {iter+1} out of {max_jumps}")

        # Create all deltas to make new points
        pas_lambda = saut[0] / ratio
        pas_gamma = saut[1] / ratio
        delta_lambda = [pas_lambda, -pas_lambda, 0]
        delta_gamma = [pas_gamma, -pas_gamma, 0]

        # Create new points
        new_points = []
        for p in points:
            new_points.append(p)
            for g in delta_gamma:
                for l in delta_lambda:
                    new_points.append([max(0, p[0] + l), max(0, p[1] + g)])

        # Compute f_scores for every point
        f_scores = []
        for p in new_points:
            lambda_, gamma = p[0], p[1]
            w_opti, _ = reg_logistic_regression(
                ytrain,
                xtrain,
                lambda_,
                w,
                max_iter,
                gamma,
            )
            y_pred = predict(xtest, w_opti, proba=True)
            f_scores.append(f1_score(ytest, y_pred))

        # if the f-score did not improve, refine by dividing the ratio
        if best_f == max(f_scores):
            ratio = ratio / 2

        best_f = max(f_scores)

        # keep only the np_points best points
        points = []
        index = f_scores.index(best_f)
        points.append(new_points[index])

        new_points.pop(index)
        f_scores.pop(index)

        for i in range(1, nb_points):
            index = f_scores.index(max(f_scores))
            points.append(new_points[index])

            new_points.pop(index)
            f_scores.pop(index)

    # Because of the way we pop/append, points is ordered by decresing f-score
    lambda_ = points[0][0]
    gamma = points[0][1]

    print(
        f"After local search, new best f-score : {best_f}, for lambda = {lambda_} and gamma = {gamma}"
    )

    return [lambda_, gamma]
