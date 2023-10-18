from sklearn.linear_model import LogisticRegression
from helpers import *
from implementations import *
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


x, x_test, y, _, test_ids = load_csv_data("dataset/")
test_ids = test_ids.astype(dtype=int)


from sklearn.model_selection import train_test_split


def create_train_test_split(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


x, x_test, y, y_test = create_train_test_split(x, y)


imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(x)
x = imp.transform(x)

imp = imp.fit(x_test)
x_test = imp.transform(x_test)
"""
fs = SelectKBest(score_func=f_classif, k=320)
X_selected = fs.fit_transform(x, y)
f = fs.get_support(1)
x = x[:, f]
x_test = x_test[:, f]
"""
print(x.shape)
print(y.shape)
print(x_test.shape)


"""
# check version number


# Keep only the meaningful columns"""
"""x, x_test, y, _, test_ids = load_csv_data("dataset/")
indexColumnsToKeep = []
mapCols = dict()
newIndex = 0
for i, label in enumerate(columns):
    if label in columnsToKeep:
        for j, col in enumerate(columnsToKeep):
            if label == col:
                indexColumnsToKeep.append(i)
                mapCols[j] = newIndex
                newIndex += 1
                break
x = x[:, indexColumnsToKeep]
indexLinesToKeep = []
for ind, line in enumerate(x):
    if np.all(np.logical_not(np.isnan(line))):
        indexLinesToKeep.append(ind)


x = x[indexLinesToKeep, :]
y = y[indexLinesToKeep]
x_test = x_test[:, indexColumnsToKeep]"""


"""
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

y = np.genfromtxt("y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0)
x = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)
x_test = np.genfromtxt("x_test_processed.csv", delimiter=" ", skip_header=0)
test_ids = np.genfromtxt("test_ids.csv", delimiter=",")
col_means = np.nanmean(x_test, axis=0)
x_test = np.where(np.isnan(x_test), col_means, x_test)
"""
print(">>> Before resample: \n --------------------------------------")
print(
    "ytrain -1: "
    + str(np.count_nonzero(y == -1))
    + "   |  ytrain 1: "
    + str(np.count_nonzero(y == 1))
    + "\n-------------------------------------\n"
)


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=steps)
x, y = pipeline.fit_resample(x, y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)
x_test = scaler.transform(x_test)

"""
from imblearn.under_sampling import RandomUnderSampler
undersampler = RandomUnderSampler(sampling_strategy="auto", random_state=42)
x, y = undersampler.fit_resample(x, y)
oversample = SMOTE(sampling_strategy="auto", random_state=42)
x, y = oversample.fit_resample(x, y)
"""
print(">>> After resample: \n --------------------------------------")
print(
    "ytrain -1: "
    + str(np.count_nonzero(y == -1))
    + "   |  ytrain 1: "
    + str(np.count_nonzero(y == 1))
    + "\n-------------------------------------"
)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


"""def tune_gradient_boosting_hyperparameters(x, y):
    # Define the hyperparameters and their possible values
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None],
    }

    # Create the Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(random_state=42)

    # Perform a grid search with cross-validation
    grid_search = RandomizedSearchCV(
        gbc, param_grid, n_iter=50, cv=5, scoring="f1", n_jobs=-1, verbose=2
    )

    # Fit the model to the data
    grid_search.fit(x, y)

    # Print the best hyperparameters and the corresponding F1 score
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best F1 Score: ", grid_search.best_score_)
    #Best Hyperparameters:  {'subsample': 1.0, 'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 4, 'learning_rate': 0.1}
    return grid_search.best_estimator_


best_gbc = tune_gradient_boosting_hyperparameters(x, y)"""

"""gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    subsample=0.1,
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_depth=4,
    learning_rate=0.1,
)"""

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def evaluate(xtrain, ytrain, xtest, ytest):
    f1_t = []
    accuracy_t = []
    for k in range(1, 300, 20):
        selector = SelectKBest(score_func=f_classif, k=k)
        x_train = selector.fit_transform(xtrain, ytrain)
        x_test = selector.transform(xtest)
        gb_classifier = LogisticRegression(max_iter=1000)
        gb_classifier.fit(x_train, ytrain)
        prediction = gb_classifier.predict(x_test)
        f1 = f1_score(ytest, prediction)
        accuracy = accuracy_score(ytest, prediction)
        f1_t.append(f1)
        accuracy_t.append(accuracy)
        print("Actuellement entrain de calculer le k numero: " + str(k))
    return f1_t, accuracy_t


f1_t, accuracy_t = evaluate(x, y, x_test, y_test)
save = [f1_t, accuracy_t]
np.savetxt("results/LogReg.txt", save)
print(save)
import matplotlib.pyplot as plt

plt.plot(range(1, 300, 20), f1_t)
plt.plot(range(1, 300, 20), accuracy_t)
# gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5)
# gb_classifier.fit(x, y)
# evaluate_model(gb_classifier, x_test, y_test)

# prediction = gb_classifier.predict(x_test)
# print("pred -1: " + str(np.count_nonzero(prediction == -1)))
# create_csv_submission(test_ids, prediction, "y_predGBC.csv")  # F1:    | Acc:"""


"""
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(x, y)
prediction = gb_classifier.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(test_ids, prediction, "y_predGBC.csv")  # F1:    | Acc:"""
"""
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(x, y)
prediction = logreg.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(
    test_ids, prediction, "y_predLOGREG.csv"
)  # F1: 0.338    | Acc: 0.848

"""
"""
# Support Vector Machines
from sklearn.svm import LinearSVC

svc = LinearSVC(max_iter=1000)
svc.fit(x, y)
prediction = svc.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(
    test_ids, prediction, "y_predSVC.csv"
)  # F1: 0.314    | Acc: 0.756 """
"""
# Decision Trees
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(x, y)
prediction = dtc.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(
    test_ids, prediction, "y_predDTC.csv"
)  # F1: 0.227     | Acc: 0.799
"""
"""
# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x, y)
prediction = rf.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(
    test_ids, prediction, "y_predRF.csv"
)  # F1: 0.226     | Acc: 0.816"""

"""# Naive Bayes
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x, y)
prediction = NB.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(
    test_ids, prediction, "y_predNB.csv"
)  # F1: 0.183     | Acc: 0.746

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
KNN.fit(x, y)
prediction = KNN.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(
    test_ids, prediction, "y_predKNN.csv"
)  # F1: 0.268     | Acc: 0.762"""
