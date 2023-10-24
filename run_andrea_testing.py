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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from helpers import *
from implementations import *

import pandas as pd

import matplotlib.pyplot as plt

# To create a submission:

x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)

X_trainF, Y_trainF, X_testF = x_train, y_train, x_test

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

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X_train)
X_trainF = imp.transform(X_trainF)
imp = imp.fit(X_test)
X_testF = imp.transform(X_testF)
from imblearn.over_sampling import BorderlineSMOTE

f1 = []
acc = []
model = []
X_train = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
X_test = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)


X_trainF = np.delete(X_trainF, [9, 11, 12, 18, 19, 22], 1)
X_testF = np.delete(X_testF, [9, 11, 12, 18, 19, 22], 1)
for i in range(20):
    over = BorderlineSMOTE(sampling_strategy=0.101)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)
    x, y = pipeline.fit_resample(X_train, Y_train)
    xf, yf = pipeline.fit_resample(X_trainF, Y_trainF)

    fs = SelectKBest(score_func=f_classif, k=150)
    x = fs.fit_transform(x, y)
    f = fs.get_support(1)
    x_test = X_test[:, f]

    fs = SelectKBest(score_func=f_classif, k=150)
    xf = fs.fit_transform(xf, yf)
    f = fs.get_support(1)
    x_testf = X_testF[:, f]

    gb_classifier = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.02,
        min_samples_split=600,
        max_depth=9,
        random_state=10,
        min_samples_leaf=30,
        max_features=19,
        subsample=0.85,
    )
    gb_classifier.fit(x, y)
    y_probabilities = gb_classifier.predict_proba(x_test)[:, 1]

    y_probabilities[y_probabilities >= 0.616177553416097] = 1
    y_probabilities[y_probabilities < 0.616177553416097] = -1
    prediction = y_probabilities
    accuracy = accuracy_score(Y_test, prediction)
    precision = precision_score(Y_test, prediction)
    recall = recall_score(Y_test, prediction)
    f1score = f1_score(Y_test, prediction)
    f1.append(f1score)
    acc.append(accuracy)
    print(str(i) + " f1: " + str(f1score))
    gb_classifier = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.02,
        min_samples_split=600,
        max_depth=9,
        random_state=10,
        min_samples_leaf=30,
        max_features=19,
        subsample=0.85,
    )
    gb_classifier.fit(xf, yf)
    y_probabilities2 = gb_classifier.predict_proba(x_testf)[:, 1]

    y_probabilities2[y_probabilities2 >= 0.616177553416097] = 1
    y_probabilities2[y_probabilities2 < 0.616177553416097] = -1
    prediction2 = y_probabilities2
    name = "TESTINGROW" + str(i) + ".csv"
    create_csv_submission(test_ids, prediction2, name)

print(f1)
print(acc)
print(np.max(f1))
# create_csv_submission(test_ids, prediction, "CeciEstUneTest18.csv")


"""
imputer = KNNImputer(n_neighbors=3)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
print("KNN Imputer done")
"""

"""
imputer = IterativeImputer(max_iter=10, random_state=0)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
print("Iterative Imputer done")
"""
"""
f1 = []
acc = []
for i in range(110, 120):
    print(str(i) + " is being started")
    fs = SelectKBest(score_func=f_classif, k=125)
    X_t = fs.fit_transform(X_train, Y_train)
    f = fs.get_support(1)
    X_t2 = X_test[:, f]
    print("K Best done")

    s_star = i * 0.001
    over = BorderlineSMOTE(sampling_strategy=s_star)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)
    X_t, Y_t = pipeline.fit_resample(X_t, Y_train.ravel())
    print("Smote done")

    gb_classifier = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.02,
        min_samples_split=600,
        max_depth=9,
        random_state=10,
        min_samples_leaf=30,
        max_features=19,
        subsample=0.85,
    )
    gb_classifier.fit(X_t, Y_t.ravel())
    y_probabilities = gb_classifier.predict_proba(X_t2)[:, 1]

    y_probabilities[y_probabilities >= 0.616177553416097] = 1
    y_probabilities[y_probabilities < 0.616177553416097] = -1
    prediction = y_probabilities
    # create_csv_submission(test_ids, prediction, "bebou.csv")
    # when splitting, calculating the score
    accuracy = accuracy_score(Y_test, prediction)
    precision = precision_score(Y_test, prediction)
    recall = recall_score(Y_test, prediction)
    f1score = f1_score(Y_test, prediction)
    f1.append(f1score)
    acc.append(accuracy)
    print(str(i) + " is done")
np.savetxt("results/f1.txt", f1)
np.savetxt("results/acc.txt", acc)

plt.plot(range(110, 120), f1)
plt.plot(range(110, 120), acc)
plt.show()
"""

"""
# Threshold tuning
print(y_probabilities)


precision, recall, thresholds = precision_recall_curve(Y_test, y_probabilities)
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()

f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(optimal_threshold)
y_predictions = (y_probabilities >= optimal_threshold).astype(int)
print(y_predictions)
# Calculate F1 score and accuracy with the optimal threshold
f1_optimal = f1_score(Y_test, y_predictions, average="micro")
accuracy_optimal = accuracy_score(Y_test, y_predictions)

print(f"Optimal F1 Score: {f1_optimal:.3f}")
print(f"Optimal Accuracy: {accuracy_optimal:.3f}")
"""

# OPTIMAL : 0.616177553416097

"""
y_probabilities[y_probabilities >= 0.616177553416097] = 1
y_probabilities[y_probabilities < 0.616177553416097] = -1
prediction = y_probabilities
create_csv_submission(test_ids, prediction, "bebou.csv")
"""
"""
#when splitting, calculating the score
accuracy = accuracy_score(Y_test, prediction)
precision = precision_score(Y_test, prediction)
recall = recall_score(Y_test, prediction)
f1score = f1_score(Y_test, prediction)

print(f"Accuracy = {accuracy}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F1 Score = {f1score}")
"""

"""
# Randomized Search CV for GBC

param_test2 = {"subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
random_search = RandomizedSearchCV(
    GradientBoostingClassifier(
        n_estimators=1600,
        learning_rate=0.005,
        min_samples_split=600,
        max_depth=9,
        random_state=10,
        min_samples_leaf=30,
        max_features=19,
        subsample=0.85
    ),
    param_test2,
    cv=5,
    scoring="f1",
    verbose=2,
    n_jobs=4,
)
random_search.fit(X_train, Y_train.ravel())
print(random_search.best_params_, random_search.best_score_)
# {'n_estimators': 80} 0.22683716015537506
# {'min_samples_split': 600, 'max_depth': 9} 0.7055290061489458
# {'min_samples_leaf': 30} 0.7013482463862419
"""
