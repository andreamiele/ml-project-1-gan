import numpy as np
from imp import *
from helpers import *
from implementations import *
from run_fonctions import *
from anova_selection import anova_f
from SMOTE import *


def transforme(X, f):
    l = []
    for i in range(len(f)):
        if f[i]:
            l.append(i)
    print(l)
    return np.transpose(np.transpose(X)[l])


def preprocessing(
    X_train,
    X_test,
    Y_train,
    Kselected,
    sampling_strat1=0.105,
    sampling_strat2=0.5,
):
    imp = SimpleImputer()
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
    ros = RandomOverSampler(sampling_strategy=sampling_strat1)
    X_balanced, Y_balanced = ros.fit_generate(X_t, Y_train)
    rus = RandomUnderSampler(sampling_strategy=sampling_strat2)
    X_t, Y_t = rus.fit_resample(X_balanced, Y_balanced)

    print(">>> After resample: \n --------------------------------------")
    print(
        "ytrain -1: "
        + str(np.count_nonzero(Y_t == -1))
        + "   |  ytrain 1: "
        + str(np.count_nonzero(Y_t == 1))
        + "\n-------------------------------------\n"
    )
    print("Over/Under sampling done")

    strat = str(int(sampling_strat1 * 1000)) + "_" + str(int(sampling_strat2 * 1000))
    fs = anova_f(X_t, Y_t, strat, k=Kselected, use_autosave=True)
    X_t = transforme(X_t, fs)
    X_t2 = transforme(X_t2, fs)
    # f = fs.get_support(1)

    print("K Best done")

    return X_t, X_t2, Y_t
