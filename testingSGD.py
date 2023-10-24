import numpy as np
from helpers import *
from implementations import *
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing

y_train = np.genfromtxt(
    "y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0
)
x_train = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)
x_test = np.genfromtxt("x_test_processed.csv", delimiter=" ", skip_header=0)
ids = np.genfromtxt("test_ids.csv", delimiter=",")


X_train, Y_train, X_test = x_train, y_train, x_test

print("data imported")


def create_train_test_split(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, Y_train, Y_test = create_train_test_split(x_train, y_train)

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train = max_abs_scaler.fit_transform(X_train)
X_test = max_abs_scaler.transform(X_test)

print("data splitted")
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
imp = imp.fit(X_test)
X_test = imp.transform(X_test)
print("data filled")


"""
def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))


def calculate_loss(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])


def calculate_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w


def calculate_hessian(y, tx, w):
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1 - pred))
    return tx.T.dot(r).dot(tx) * (1 / y.shape[0])


def logistic_regression(y, tx, w):
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w, gamma):
    loss, gradient, hessian = logistic_regression(y, tx, w)
    w -= gamma * np.linalg.solve(hessian, gradient)
    return loss, w


def LogRegNwt(y, x):
    # init parameters
    max_iter = 100
    threshold = 1e-2
    lambda_ = 0.1
    gamma = 1.0
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))
    for iter in range(max_iter):
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w


def predict(X, w):
    model = np.dot(X, w)
    predictions = sigmoid(model)
    return predictions


def LogReg(y, x):
    # init parameters
    max_iter = 1000
    threshold = 1e-2
    gamma = 0.5
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))
    for iter in range(max_iter):
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w


w = LogReg(Y_train, X_train)
w2 = LogRegNwt(Y_train, X_train)
Y_pred = predict(X_test, w)
Y_pred2 = predict(X_test, w2)

print(Y_pred)
plt.hist(Y_pred)
print(Y_pred2)
plt.hist(Y_pred2)
plt.show()
Y_pred = np.where(Y_pred <= 0.5, -1, 1)
Y_pred2 = np.where(Y_pred2 <= 0.5, -1, 1)
# create_csv_submission(ids, y_pred, "y_pred3.csv")

# Evaluate the model's performance (e.g., F1 score)
from sklearn.metrics import f1_score

f1 = f1_score(Y_test, Y_pred, average="micro")
f1_2 = f1_score(Y_test, Y_pred2, average="micro")
print(f"F1 Score: {f1:.2f}")
print(f"F1 Score: {f1_2:.2f}")
"""

clf = SGDClassifier(loss="log_loss", penalty="l2")
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
Y_pred = np.where(Y_pred == 0, -1, 1)
print("F1: {:.2f}".format(f1_score(Y_test, Y_pred)))

clf = LogisticRegression(max_iter=1000, solver="liblinear")
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
Y_pred = np.where(Y_pred == 0, -1, 1)
print("F1: {:.2f}".format(f1_score(Y_test, Y_pred)))

model = LogisticRegression(max_iter=1000, solver="liblinear")
y_proba = cross_val_predict(model, X_train, Y_train, cv=5, method="predict_proba")[:, 1]

# Define a range of thresholds to test
thresholds = np.linspace(0, 1, 100)

# Initialize variables to store the best threshold and corresponding F1 score
best_threshold = 0
best_f1 = 0

# Iterate over the thresholds and calculate F1 score for each
for threshold in thresholds:
    y_pred = np.where(y_proba >= threshold, -1, 1)
    f1 = f1_score(Y_train, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
print(best_threshold)
# Train the model on the entire training set with the best threshold

y_train = np.genfromtxt(
    "y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0
)
x_train = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)
x_test = np.genfromtxt("x_test_processed.csv", delimiter=" ", skip_header=0)
ids = np.genfromtxt("test_ids.csv", delimiter=",")


X_train, Y_train, X_test = x_train, y_train, x_test

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train = max_abs_scaler.fit_transform(X_train)
X_test = max_abs_scaler.transform(X_test)

print("data splitted")
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
imp = imp.fit(X_test)
X_test = imp.transform(X_test)
print("data filled")


model.fit(X_train, Y_train)
y_probabilities_test = model.predict_proba(X_test)[:, 1]
print(y_probabilities_test.shape)
y_pred_test = np.where(y_probabilities_test >= best_threshold, -1, 1)
print(y_pred_test.shape)
create_csv_submission(ids, y_pred_test, "bouboi.csv")
print("F1: {:.2f}".format(f1_score(Y_test, y_pred_test)))
