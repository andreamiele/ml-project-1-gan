import numpy as np
from imp import *
from helpers import *
from implementations import *
from run_fonctions import *
from validation import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from SMOTE import *


# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline


def preprocessing(X_train, X_test, Y_train, sampling_strat, Kselected):
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp = imp.fit(X_test)
    X_test = imp.transform(X_test)

    X_t = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
    X_t2 = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)

    print(">>> Before resample: \n --------------------------------------")
    print(
        "ytrain -1: "
        + str(np.count_nonzero(Y_train == -1))
        + "   |  ytrain 1: "
        + str(np.count_nonzero(Y_train == 1))
        + "\n-------------------------------------\n"
    )
    ros = RandomOverSampler(sampling_strategy=0.15)
    X_balanced, Y_balanced = ros.fit_generate(X_t, Y_train)
    rus = RandomUnderSampler(sampling_strategy=0.8)
    X_t, Y_t = rus.fit_resample(X_balanced, Y_balanced)

    print(">>> Before resample: \n --------------------------------------")
    print(
        "ytrain -1: "
        + str(np.count_nonzero(Y_t == -1))
        + "   |  ytrain 1: "
        + str(np.count_nonzero(Y_t == 1))
        + "\n-------------------------------------\n"
    )
    print("Over/Under sampling done")

    fs = SelectKBest(score_func=f_classif, k=Kselected)
    X_t = fs.fit_transform(X_t, Y_t)
    X_t2 = fs.transform(X_t2)
    f = fs.get_support(1)
    print("K Best done")
    return X_t, X_t2, Y_t
