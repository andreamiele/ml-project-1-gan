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


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, x):
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

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for iteration in range(self.num_iterations):
            for i in range(m):
                # Randomly select a data point (stochastic gradient descent)
                random_index = np.random.randint(m)
                xi = X[random_index, :]
                yi = y[random_index]

                # Calculate the predicted probability
                z = np.dot(xi, self.theta)
                h = self.sigmoid(z)

                # Compute the gradient and update the weights
                gradient = xi * (h - yi)
                self.theta -= self.learning_rate * gradient

    def predict(self, X, proba=False):
        # Calculate predicted probabilities and return binary predictions
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        if proba == False:
            return (h >= 0.5).astype(int)
        else:
            return h


# To create a submission:

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)

X_train, Y_train, X_test = x_train, y_train, x_test

# For splitting data:


def create_train_test_split(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, Y_train, Y_test = create_train_test_split(x_train, y_train)


imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
imp = imp.fit(X_test)
X_test = imp.transform(X_test)
from imblearn.over_sampling import BorderlineSMOTE

X_train = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
X_test = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)
over = BorderlineSMOTE(sampling_strategy=0.105)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=steps)
x, y = pipeline.fit_resample(X_train, Y_train)

fs = SelectKBest(score_func=f_classif, k=30)
x = fs.fit_transform(x, y)
x_test = fs.transform(X_test)
f = fs.get_support(1)


learning_rates = [0.01, 0.1, 0.5]
num_iterations = [100, 500, 1000]

best_f1_score = 0
best_hyperparameters = None

# Perform grid search over hyperparameters
for learning_rate in learning_rates:
    for iterations in num_iterations:
        print("currently testing: " + str(learning_rate) + " and " + str(iterations))
        model = LogisticRegressionSGD(
            learning_rate=learning_rate, num_iterations=iterations
        )
        model.fit(x, y)
        y_pred = model.predict(x_test)
        y_pred[y_pred == 0] = -1
        f1 = f1_score(Y_test, y_pred, average="micro")
        print(f1)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_hyperparameters = (learning_rate, iterations)

print("Best Hyperparameters (Learning Rate, Num Iterations):", best_hyperparameters)
print("Best F1 Score:", best_f1_score)
