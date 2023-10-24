#!/usr/bin/env python3
# -*- coding: utf-8 -*-
### >>> IMPORT
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
from anova_selection import anova_f
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from helpers import *
from implementations import *

import pandas as pd

import matplotlib.pyplot as plt
import progressbar
#from check.misc import bar_widgets


# To create a submission:

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)

X_train, Y_train, X_test = x_train, y_train, x_test

# For splitting data:

"""
def create_train_test_split(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, Y_train, Y_test = create_train_test_split(x_train, y_train)
"""


import copy
import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true.dot(np.log(y_pred + 1e-9))
        y_one_loss = (1 - y_true).dot(np.log(1 - y_pred + 1e-9))
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def predict_proba(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)


imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
imp = imp.fit(X_test)
X_test = imp.transform(X_test)
from imblearn.over_sampling import BorderlineSMOTE

X_t = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
X_t2 = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)

over = BorderlineSMOTE(sampling_strategy=0.104)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=steps)
X_t, Y_t = pipeline.fit_resample(X_t, Y_train.ravel())
print("Smote done")

fs = anova_f(X_t, Y_t, k = 20)
X_t = fs.fit_transform(X_t, Y_t)
X_t2 = fs.transform(X_t2)
f = fs.get_support(1)
print("K Best done")

lr = LogisticRegression()
lr.fit(X_t, Y_t, epochs=50)
pred = lr.predict_proba(X_t2)
print(pred)