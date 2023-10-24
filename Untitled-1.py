"""#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: Untitled-1.py
# Created: Friday, 20th October 2023 5:32:22 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Sunday, 22nd October 2023 7:14:09 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023 Your Company
#
#  ==============================================================================
# %%
import pandas as pd

df = pd.read_csv("dataset/x_train.csv")
df2 = pd.read_csv("dataset/x_test.csv")
print(df.head())

# %%
print(df.info())

# %%
print(df.columns)

# %%
print(df.describe())

# %%
df = df[
    [
        "BPMEDS",
        "_RFHYPE5",
        "EMPLOY1",
        "_TOTINDA",
        "_BMI5",
        "_FRTLT1",
        "_VEGLT1",
        "_RFDRHV5",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "MENTHLTH",
        "PHYSHLTH",
        "EDUCA",
        "INCOME2",
        "GENHLTH",
        "_RFCHOL",
        "_AGE80",
        "SEX",
        "CVDSTRK3",
        "DIFFWALK",
        "MAXVO2_",
        "CHCCOPD1",
        "PNEUVAC3",
        "QLACTLM2",
        "HAREHAB1",
        "TOLDHI2",
        "SMOKE100",
        "CHOLCHK",
        "CHCKIDNY",
        "CVDASPRN",
        "DIABETE3",
        "ASPUNSAF",
        "_SMOKER3",
        "_DRDXAR1",
        "_RFSEAT2",
        "VIPRFVS2",
        "USEEQUIP",
        "INTERNET",
        "DRADVISE",
        "PERSDOC2",
        "BPHIGH4",
    ]
]

df2 = df2[
    [
        "BPMEDS",
        "_RFHYPE5",
        "EMPLOY1",
        "_TOTINDA",
        "_BMI5",
        "_FRTLT1",
        "_VEGLT1",
        "_RFDRHV5",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "MENTHLTH",
        "PHYSHLTH",
        "EDUCA",
        "INCOME2",
        "GENHLTH",
        "_RFCHOL",
        "_AGE80",
        "SEX",
        "CVDSTRK3",
        "DIFFWALK",
        "MAXVO2_",
        "CHCCOPD1",
        "PNEUVAC3",
        "QLACTLM2",
        "HAREHAB1",
        "TOLDHI2",
        "SMOKE100",
        "CHOLCHK",
        "CHCKIDNY",
        "CVDASPRN",
        "DIABETE3",
        "ASPUNSAF",
        "_SMOKER3",
        "_DRDXAR1",
        "_RFSEAT2",
        "VIPRFVS2",
        "USEEQUIP",
        "INTERNET",
        "DRADVISE",
        "PERSDOC2",
        "BPHIGH4",
    ]
]

# %%
df = df.fillna(df.mean())
df2 = df2.fillna(df.mean())

x = [
    "BPMEDS",
    "_RFHYPE5",
    "EMPLOY1",
    "_TOTINDA",
    "_BMI5",
    "_FRTLT1",
    "_VEGLT1",
    "_RFDRHV5",
    "HLTHPLN1",
    "MEDCOST",
    "GENHLTH",
    "MENTHLTH",
    "PHYSHLTH",
    "EDUCA",
    "INCOME2",
    "GENHLTH",
    "_RFCHOL",
    "_AGE80",
    "SEX",
    "CVDSTRK3",
    "DIFFWALK",
    "MAXVO2_",
    "CHCCOPD1",
    "PNEUVAC3",
    "QLACTLM2",
    "HAREHAB1",
    "TOLDHI2",
    "SMOKE100",
    "CHOLCHK",
    "CHCKIDNY",
    "CVDASPRN",
    "DIABETE3",
    "ASPUNSAF",
    "_SMOKER3",
    "_DRDXAR1",
    "_RFSEAT2",
    "VIPRFVS2",
    "USEEQUIP",
    "INTERNET",
    "DRADVISE",
    "PERSDOC2",
    "BPHIGH4",
]

# %%
one_hot_encoded_data = pd.get_dummies(
    df,
    columns=[
        "_RFHYPE5",
        "TOLDHI2",
        "_BMI5",
        "SMOKE100",
        "CVDSTRK3",
        "DIABETE3",
        "_TOTINDA",
        "_FRTLT1",
        "_VEGLT1",
        "_RFDRHV5",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "MENTHLTH",
        "PHYSHLTH",
        "DIFFWALK",
        "SEX",
        "EDUCA",
        "INCOME2",
    ],
)

one_hot_encoded_data2 = pd.get_dummies(
    df2,
    columns=[
        "_RFHYPE5",
        "TOLDHI2",
        "_BMI5",
        "SMOKE100",
        "CVDSTRK3",
        "DIABETE3",
        "_TOTINDA",
        "_FRTLT1",
        "_VEGLT1",
        "_RFDRHV5",
        "HLTHPLN1",
        "MEDCOST",
        "GENHLTH",
        "MENTHLTH",
        "PHYSHLTH",
        "DIFFWALK",
        "SEX",
        "EDUCA",
        "INCOME2",
    ],
)

print(one_hot_encoded_data)

# %%
# one_hot_encoded_data.to_csv("dataset/xPRE.csv", index=False)
# one_hot_encoded_data2.to_csv("dataset/xTPre.csv", index=False)

# %%
"""
from sklearn.linear_model import LogisticRegression
from helpers import *
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    get_scorer_names,
)
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest

# %%
x, x_test, y, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)
x = pd.read_csv("dataset/xPRE.csv")
x_test = pd.read_csv("dataset/xTPRE.csv")

"""
from sklearn.model_selection import train_test_split


def create_train_test_split(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


x, x_test, y, y_test = create_train_test_split(x, y)
"""
print("loaded")

fs = SelectKBest(score_func=f_classif, k=23)
x = fs.fit_transform(x, y)
x_test = fs.transform(x_test)
f = fs.get_support(indices=True)
# %%
over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.4)
steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=steps)
x, y = pipeline.fit_resample(x, y)
print("smote")
# %%

from sklearn.linear_model import SGDClassifier


clf = SGDClassifier(
    loss="log_loss",
    penalty=None,
    alpha=0.0001,
    max_iter=340,
    tol=0.00001,
    shuffle=True,
    verbose=0,
    random_state=1,
    learning_rate="optimal",
    eta0=2,
)
clf.fit(x, y)

print("fitted")


# predict = clf.predict(x_test)
proba = clf.predict_proba(x_test)[:, 1]
print(np.quantile(proba, 0.854))
plt.hist(proba)
plt.show()
proba[proba >= np.quantile(proba, 0.854)] = 1
proba[proba < np.quantile(proba, 0.854)] = -1
print("pred -1: " + str(np.count_nonzero(proba == -1)))
create_csv_submission(test_ids, proba, "y_predSGD_OHE.csv")  # F1: 0.338    | Acc: 0.848

"""
f1T = []
accT = []
for i in range(5, 100, 2):
    x_p = x
    x_test_p = x_test
    y_p = y
    fs = SelectKBest(score_func=f_classif, k=i)
    X_selected = fs.fit_transform(x_p, y_p)
    f = fs.get_support(1)
    x_p = fs.fit_transform(x_p, y_p)
    x_test_p = fs.transform(x_test_p)
    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.4)
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)
    x_p, y_p = pipeline.fit_resample(x_p, y_p)
    clf = SGDClassifier(
        loss="log_loss",
        penalty=None,
        alpha=0.0001,
        max_iter=340,
        tol=0.00001,
        shuffle=True,
        verbose=0,
        random_state=1,
        learning_rate="optimal",
        eta0=2,
    )
    clf.fit(x_p, y_p)
    proba = clf.predict_proba(x_test_p)[:, 1]
    print(np.quantile(proba, 0.854))
    proba[proba >= np.quantile(proba, 0.854)] = 1
    proba[proba < np.quantile(proba, 0.854)] = -1
    f1 = f1_score(y_test, proba)
    f1T.append(f1)
    acc = accuracy_score(y_test, proba)
    accT.append(acc)
    print("F1 :" + str(f1) + " Acc :" + str(acc))
print(f1T)
print(accT)
plt.plot(range(5, 100, 2), f1T)
plt.plot(range(5, 100, 2), accT)
plt.show()
"""

"""

print("pred -1: " + str(np.count_nonzero(predict == -1)))
create_csv_submission(
    test_ids, predict, "y_predSGDLOGRED.csv"
)  # F1: 0.338    | Acc: 0.848
#0.854
"""
"""
f1T = []
accT = []
maxi = 0
max_idx = 0
for i in range(200, 950, 1):
    p = i * 0.001
    print("Testing :" + str(p))
    probi = clf.predict_proba(x_test)[:, 1]
    q = np.quantile(probi, p)
    probi[probi >= q] = 1
    probi[probi < q] = -1
    print("pred -1: " + str(np.count_nonzero(probi == -1)))
    f1 = f1_score(y_test, probi)
    if f1 > maxi:
        maxi = f1
        max_idx = p
    f1T.append(f1)
    acc = accuracy_score(y_test, probi)
    accT.append(acc)
    print("F1 :" + str(f1) + " Acc :" + str(acc))
print("max is for : " + str(max_idx))
print(f1T)
print(accT)
plt.plot(range(300, 950, 1), f1T)
plt.plot(range(300, 950, 1), accT)
plt.show()
"""
# %%
"""
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import parfit.parfit as pf

grid = {
    "alpha": [1e-4],  # learning rate
    "max_iter": [340],  # number of epochs
    "loss": ["log_loss"],
    "learning_rate": ["optimal"],  # logistic regression,
    "penalty": ["l2"],
    "eta0": [0, 1, 2, 3, 4, 5, 10, 20, 200],
}
paramGrid = ParameterGrid(grid)

bestModel, bestScore, allModels, allScores = pf.bestFit(
    SGDClassifier,
    paramGrid,
    x,
    y,
    x_test,
    y_test,
    metric=f1_score,
    greater_is_better=True,
    scoreLabel="F1",
)


print(bestModel, bestScore)
"""
