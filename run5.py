import numpy as np
from imp import *
from helpers import *
import matplotlib.pyplot as plt
from run_fonctions import *
from validation import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import precision_recall_curve
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    get_scorer_names,
)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from helpers import *
from implementations import *

import pandas as pd

import matplotlib.pyplot as plt
import progressbar
from check.misc import bar_widgets

from new_preprocessing import *

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)


def create_train_test_split(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


X_train, Y_train, X_test = x_train, y_train, x_test

# For splitting data:


def predict_labels(weights, x):
    """Generates class predictions given weights, and a test data matrix"""
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    y_pred = np.dot(tx, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def predict_labels2(weights, x):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(x, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def predict_labels_threshold(weights, x, threshold):
    """Generates class predictions given weights, and a test data matrix"""
    # tx = np.c_[np.ones((x.shape[0], 1)), x]
    # y_pred = np.dot(tx, weights)
    y_pred = np.dot(x, weights)
    thresholds = np.quantile(y_pred, threshold)
    y_pred[np.where(y_pred <= thresholds)] = -1
    y_pred[np.where(y_pred > thresholds)] = 1

    return y_pred


def predict_proba(x, w):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probabilities = sigmoid(tx.dot(w))
    return [1 if p > 0.5 else -1 for p in probabilities]


def predict_proba_threshold(x, w, threshold):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probabilities = sigmoid(tx.dot(w))
    return [1 if p > threshold else -1 for p in probabilities]


"""
X_train, X_test, Y_train, Y_test = create_train_test_split(x_train, y_train)


thresholds = range(20, 600, 20)
Kbest = range(20, 60, 5)
best = 0
best_param = [0, 0, 0, 0]

for threshold_ in thresholds:
    for k in Kbest:
        threshold = 0.001 * threshold_
        print(f">>> Training for:\n threshold: {threshold} | K: {k} ")
        X_t, X_t2, yb_train = preprocessing(X_train, X_test, Y_train, 0.1, k)
        x_train = standardize(X_t)
        x_test = standardize(X_t2)
        w, loss = run_logistic_regression(yb_train, x_train)
        y_pred = predict_proba_threshold(x_test, w, threshold)

        f1 = f1_score(Y_test, y_pred)
        if f1 > best:
            best = f1
            best_param = [threshold, k]
        print(f"f1= {f1}")
        print("Done \n -------------")
print("best")
print(best)
print(best_param)

"""


def main():

    ###################### FEATURE PROCESSING ##################
    print("Feature processing")

    # Remove selected features
    # input_data_train, input_data_test = removecols(input_data_train, input_data_test, [14,15,17,18,24,25,27,28])
    X_t, X_t2, yb_train = preprocessing(X_train, X_test, Y_train, 0.1, 150)
    # Standardize and sentralize data
    x_train = standardize(X_t)
    x_test = standardize(X_t2)

    # Build model test data
    # y_test, tx_test = build_model_data(x_test,yb_test)

    ###################### RUN FUNCTIONS #####################
    w, loss = run_gradient_descent(yb_train, x_train)
    # Build model test data must be applied when running run_gradient_descent
    # y_test, tx_test = build_model_data(x_test, yb_test)
    y_pred = predict_labels(w, x_test)
    create_csv_submission(test_ids, y_pred, "results/baseline/results_GD150.csv")
    print("GD Finished")
    w, loss = run_stochastic_gradient_descent(yb_train, x_train)
    # Build model test data must be applied when running run_stochastic_gradient_descent
    # y_test, tx_test = build_model_data(x_test,yb_test)
    y_pred = predict_labels(w, x_test)
    create_csv_submission(test_ids, y_pred, "results/baseline/results_SGD150.csv")
    print("SGD Finished")
    w, loss, degree = run_least_square(yb_train, x_train)
    # Build model poly data, has to be done wen running run_least_square
    tx_test = build_poly(x_test, degree)
    y_pred = predict_labels2(w, tx_test)
    create_csv_submission(test_ids, y_pred, "results/baseline/results_LS150.csv")
    print("LS Finished")
    w, loss, degree = run_ridge_regression(yb_train, x_train, 10e-5)
    # Build poly data, has to be done when running run_ridge_regression
    tx_test = build_poly(x_test, degree)
    y_pred = predict_labels2(w, tx_test)
    create_csv_submission(test_ids, y_pred, "results/baseline/results_RR150.csv")
    print("RR Finished")
    w, loss = run_logistic_regression(yb_train, x_train, 0.01, 1000)
    # Build model test data must be applied when running run_logistic_regression
    # y_test, tx_test = build_model_data(x_test, y_test)
    y_pred = predict_proba(x_test, w)
    create_csv_submission(test_ids, y_pred, "results/baseline/results_LR150.csv")
    print("LR Finished")
    w, loss = run_reg_logistic_regression(yb_train, x_train, 0.01, 0.0001, 1000)
    # Build model test data must be applied when running run_reg_logistic_regression
    # y_test, tx_test = build_model_data(x_test,yb_test)
    y_pred = predict_proba(x_test, w)
    create_csv_submission(test_ids, y_pred, "results/baseline/results_RLR150.csv")
    print("RLR Finished")
    # When performing stacking, the predicted labels are given directly #
    # y_pred = stacking(yb_train,x_train,yb_test,x_test)

    ###################### VALIDATIONS ########################
    # gradientdescent_gamma(yb_train, x_train)

    # stochastic_gradientdescent_gamma(yb_train, x_train)

    # leastsquares_degree(yb_train, x_train)

    # ridgeregression_lambda(yb_train, x_train)

    # ridgeregression_degree_lambda(yb_train, x_train)

    # logregression_gamma(yb_train, x_train)

    # logregression_gamma_degree(yb_train, x_train)

    # reglogregression_gamma_lambda(yb_train, x_train)

    # stacking_crossvalidation(yb_train, x_train)

    ################## MAKE PREDICTIONS #####################

    # y_pred = predict_proba_threshold(x_test, w, 0.50)
    # y_pred = predict_labels_threshold(w, tx_test, 0.50)
    # create_csv_submission(test_ids, y_pred, "resultsLR.csv")
    print("Finished")
    return 0


### Run main function
main()
