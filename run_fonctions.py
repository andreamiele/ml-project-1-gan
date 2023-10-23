import numpy as np
from imp import *
from helpers import *
import matplotlib.pyplot as plt


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


####################### RUN FUNCTIONS #####################################
# The run functions runs the various functions given the parameters given #
# at the beginning of every functions. These parameters were find through #
# optimization, but can be changed                                        #
def run_gradient_descent(y, x):
    max_iters = 300
    gamma = 0.2
    y, tx = build_model_data(x, y)
    initial_w = np.zeros(tx.shape[1])
    gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)

    return gd_w, gd_loss


def run_stochastic_gradient_descent(y, x):
    y, tx = build_model_data(x, y)
    max_iters = 100
    gamma = 0.0001
    initial_w = np.zeros(tx.shape[1])
    sgd_w, sgd_loss = least_squares_SGD(y, tx, initial_w, max_iters, gamma)

    return sgd_w, sgd_loss


def run_least_square(y, x):
    degree = 10
    tx = build_poly(x, degree)
    ls_w, ls_loss = least_squares(y, tx)

    return ls_w, ls_loss, degree


def run_ridge_regression(y, x, lambda_=10e-2, degree=3):
    tx = build_poly(x, degree)
    rr_w, rr_loss = ridge_regression(y, tx, lambda_)

    return rr_w, rr_loss, degree


def run_logistic_regression(y, x, gamma=0.1, max_iters=100):
    y, tx = build_model_data(x, y)
    initial_w = np.zeros((tx.shape[1], 1))
    y = np.expand_dims(y, axis=1)
    gamma = gamma
    max_iters = max_iters
    lr_w, lr_loss = logistic_regression(y, tx, initial_w, max_iters, gamma)

    return lr_w, lr_loss


def run_reg_logistic_regression(y, x):
    y, tx = build_model_data(x, y)
    initial_w = np.zeros((tx.shape[1], 1))
    y = np.expand_dims(y, axis=1)
    gamma = 0.0004
    lambda_ = 0.0001
    max_iters = 100
    rlr_w, rlr_loss = reg_logistic_regression(
        y, tx, lambda_, initial_w, max_iters, gamma
    )

    return rlr_w, rlr_loss


############################# STACKING ####################################
# stacking uses the same functions as implemented above, but to be        #
# compatibale with the format needed in order to run stacking() some of   #
# of the functions has been altered some                                  #


def build_poly_row(row, degree):
    # polynomial basis functions for input data row, for j=0 up to        #
    # j = degree                                                          #
    aug_row = np.ones(1)
    for d in range(1, degree + 1):
        aug_row = np.concatenate((aug_row, np.power(row, d)))

    return aug_row


def predict_labels_row(weights, data):
    # Generates class predictions given weights, and a test data row      #
    y_pred = np.dot(data, weights)
    if y_pred <= 0:
        y_pred = -1
    else:
        y_pred = 1

    return y_pred


def gradient_descent_model(train):
    max_iters = 300
    gamma = 0.2
    y = train[:, [0]]
    y = np.squeeze(np.asarray(y))
    x = np.delete(train, 0, axis=1)
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    initial_w = np.zeros(tx.shape[1])
    gd_w, gd_loss = least_squares_GD(y, tx, initial_w, max_iters, gamma)

    return gd_w


def least_square_model(train):
    degree = 10
    y = train[:, [0]]
    y = np.squeeze(np.asarray(y))
    x = np.delete(train, 0, axis=1)
    tx = build_poly(x, degree)
    ls_w, ls_loss = least_squares(y, tx)

    return ls_w


def ridge_regression_model(train):
    degree = 10
    lambda_ = 0.0001
    y = train[:, [0]]
    y = np.squeeze(np.asarray(y))
    x = np.delete(train, 0, axis=1)
    tx = build_poly(x, degree)
    rr_w, rr_loss = ridge_regression(y, tx, lambda_)

    return rr_w


def logistic_regression_model(train):
    degree = 1
    gamma = 0.1
    max_iters = 300
    y = [i[-1] for i in train]
    y = np.squeeze(np.asarray(y))
    x = np.delete(train, -1, axis=1)
    tx = build_poly(x, degree)
    initial_w = np.zeros((tx.shape[1], 1))
    y = np.expand_dims(y, axis=1)
    lr_w, lr_loss = logistic_regression(y, tx, initial_w, max_iters, gamma)

    return lr_w


def gradient_descent_predict(w, row):
    row = np.delete(row, 0)
    row_tilde = np.insert(row, 0, 1)
    row_tilde = np.transpose(row_tilde)
    y_pred = predict_labels_row(w, row_tilde)

    return y_pred


def least_square_predict(w, row):
    degree = 10
    x_row = np.delete(row, 0)
    x_row = np.transpose(x_row)
    augmented_row = build_poly_row(x_row, degree)
    y_pred = predict_labels_row(w, augmented_row)

    return y_pred


def ridge_regression_predict(w, row):
    degree = 10
    x_row = np.delete(row, 0)
    x_row = np.transpose(x_row)
    augmented_row = build_poly_row(x_row, degree)
    y_pred = predict_labels_row(w, augmented_row)

    return y_pred


def logistic_regression_predict(w, row):
    degree = 1
    x_row = np.delete(row, -1)
    x_row = np.transpose(x_row)
    augmented_row = build_poly_row(x_row, degree)
    y_pred = predict_labels_row(w, augmented_row)

    return y_pred


def to_stacked_row(models, predict_list, row):
    # Make predictions with sub-models and construct a new stacked row    #
    stacked_row = list()
    for i in range(len(models)):
        prediction = predict_list[i](models[i], row)
        stacked_row.append(prediction)
    stacked_row.append(row[0])
    rowlist = row[1 : len(row) - 1].tolist()
    # if the last model should only train on the predictions of the       #
    # others return only stacked_row                                      #

    return stacked_row  # rowlist + stacked_row


def stacking(y, x, y_test, x_test):
    train = np.c_[y.T, x]
    test = np.c_[y_test.T, x_test]
    model_list = [gradient_descent_model, least_square_model, ridge_regression_model]
    predict_list = [
        gradient_descent_predict,
        least_square_predict,
        ridge_regression_predict,
    ]
    models = list()
    # creates models from the various methods based on the training set
    for i in range(len(model_list)):
        model = model_list[i](train)
        models.append(model)
    stacked_dataset = list()
    print("Stacking: Finished creating models")
    for row in train:
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
    # creates a model with logistic regression based on the other
    # models predictions
    stacked_model = logistic_regression_model(stacked_dataset)
    print(
        "Stacking: Finished with logistic regression on the other models' predictions"
    )
    predictions = list()
    for row in test:
        # creates predictions based on the test-set
        stacked_row = to_stacked_row(models, predict_list, row)
        stacked_dataset.append(stacked_row)
        prediction = logistic_regression_predict(stacked_model, stacked_row)
        predictions.append(prediction)
    print("Stacking: Finished creating predictions")

    return predictions
