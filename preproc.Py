#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: Untitled-1.py
# Created: Friday, 20th October 2023 5:32:22 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Sunday, 22nd October 2023 2:54:03 pm
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
    ]
]

df2 = df2[
    [
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
    ]
]

# %%
df = df.fillna(df.mean())
df2 = df2.fillna(df.mean())

print(df)
print(df2)


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

# %%
x, x_test, y, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)


def create_train_test_split(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


x, x_test, y, y_test = create_train_test_split(x_train, y_train)

x = one_hot_encoded_data
x_test = one_hot_encoded_data2
print("loaded")


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
    max_iter=400,
    tol=0.00001,
    shuffle=True,
    verbose=0,
    random_state=1,
    learning_rate="optimal",
    eta0=0.0,
)
clf.fit(x, y)

print("fitted")
predict = clf.predict(x_test)
print("pred -1: " + str(np.count_nonzero(predict == -1)))
create_csv_submission(
    test_ids, predict, "y_predSGDLOGRED.csv"
)  # F1: 0.338    | Acc: 0.848


# %%
