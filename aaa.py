from sklearn.linear_model import LogisticRegression
from helpers import *
from implementations import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.impute import SimpleImputer
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
from SMOTE import *

x, x_test, y, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)

print(x[:, 223])
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(x)
x = imp.transform(x)
imp = imp.fit(x_test)
x_test = imp.transform(x_test)
y[y == -1] = 0

print(">>> Before resample: \n --------------------------------------")
print(
    "ytrain -1: "
    + str(np.count_nonzero(y == 0))
    + "   |  ytrain 1: "
    + str(np.count_nonzero(y == 1))
    + "\n-------------------------------------\n"
)
smote = SMOTE(k=5, dims=np.shape(x)[1], sampling_strategy=0.1)
X_balanced, y = smote.fit_generate(x, y)
rus = RandomUnderSampler(sampling_strategy=0.5)
X_balanced, y = rus.fit_resample(x, y)
print(">>> After resample: \n --------------------------------------")
print(
    "ytrain -1: "
    + str(np.count_nonzero(y == 0))
    + "   |  ytrain 1: "
    + str(np.count_nonzero(y == 1))
    + "\n-------------------------------------"
)
