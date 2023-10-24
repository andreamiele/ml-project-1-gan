# Import
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Import helper functions
from helpers import *
from implementations import *
from run_fonctions import *
from validation import *
from imp import *
from preprocessing import *


def create_train_test_split(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


"""
x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)
X_train, Y_train, X_test = x_train, y_train, x_test
"""
# For splitting data:


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)

    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def predict_labels_threshold(weights, data, threshold):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    thresholds = np.quantile(y_pred, threshold)
    y_pred[np.where(y_pred <= thresholds)] = -1
    y_pred[np.where(y_pred > thresholds)] = 1

    return y_pred


def predict_proba(x, w):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probabilities = sigmoid(tx.dot(w))
    return [1 if p > 0.1 else -1 for p in probabilities]


def predict_proba_threshold(x, w, threshold):
    tx = np.c_[np.ones((x.shape[0], 1)), x]
    probabilities = sigmoid(tx.dot(w))
    return [1 if p > threshold else -1 for p in probabilities]

def predict(weights,data,threshold=None,proba=False):
    if proba:
        
    else:
    
    if threshold==None:
        return [1 if prediction > 0.5 else -1 for prediction in predictions]
    else:
        return [1 if prediction > threshold else -1 for prediction in predictions]

"""
X_train, X_test, Y_train, Y_test = create_train_test_split(x_train, y_train)


x_train, x_test, yb_train = preprocessing(X_train, X_test, Y_train, 0.1, 30)
"""


def main(training=False):
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
    test_ids = test_ids.astype(dtype=int)

    if training:
        x_train, x_test, y_train, y_test = create_train_test_split(x_train, y_train)

    x_train, x_test, yb_train = preprocessing(x_train, x_test, y_train, 150, 0.15, 0.8)

    # Standardize and sentralize data
    x_train2 = standardize(x_train)
    x_test2 = standardize(x_test)

    # Build model test data
    # y_test, tx_test = build_model_data(x_test,yb_test)

    ###################### RUN FUNCTIONS #####################
    # w, loss = run_gradient_descent(yb_train, x_train)
    # Build model test data must be applied when running run_gradient_descent
    # y_test, tx_test = build_model_data(x_test,yb_test)

    # w, loss = run_stochastic_gradient_descent(yb_train, x_train)
    # Build model test data must be applied when running run_stochastic_gradient_descent
    # y_test, tx_test = build_model_data(x_test,yb_test)

    # w, loss, degree = run_least_square(yb_train,x_train)
    # Build model poly data, has to be done wen running run_least_square
    # tx_test = build_poly(x_test,degree)

    w, loss, degree = run_ridge_regression(yb_train, x_train2)
    # Build poly data, has to be done when running run_ridge_regression
    tx_test = build_poly(x_test2, degree)

    # w, loss = run_logistic_regression(yb_train, x_train)
    # Build model test data must be applied when running run_logistic_regression
    # y_test, tx_test = build_model_data(x_test, y_test)

    # w, loss = run_reg_logistic_regression(yb_train, x_train)
    # Build model test data must be applied when running run_reg_logistic_regression
    # y_test, tx_test = build_model_data(x_test,yb_test)

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

    y_pred = predict_labels_threshold(w, tx_test, 0.87)
    if training:
        f1 = f1_score(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        print("f1: " + str(f1))
        print("acc: " + str(accuracy))
    else:
        create_csv_submission(test_ids, y_pred, "resultat2.csv")
    print("Finished")
    return 0


### Run main function
main()
