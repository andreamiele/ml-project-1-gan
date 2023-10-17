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

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(x)
x = imp.transform(x)

imp = imp.fit(x_test)
x_test = imp.transform(x_test)

fs = SelectKBest(score_func=f_classif, k=40)
X_selected = fs.fit_transform(x, y)
f = fs.get_support(1)
x = x[:, f]
x_test = x_test[:, f]

print(x.shape)
print(y.shape)
print(x_test.shape)
'''
columnsToKeep = [
    "_RFHYPE5",
    "TOLDHI2",
    "_CHOLCHK",
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
    "_AGEG5YR",
    "EDUCA",
    "INCOME2",
]

columns = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "NUMADULT",
    "NUMMEN",
    "NUMWOMEN",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "HHADULT",
    "GENHLTH",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "BPHIGH4",
    "BPMEDS",
    "BLOODCHO",
    "CHOLCHK",
    "TOLDHI2",
    "CVDSTRK3",
    "ASTHMA3",
    "ASTHNOW",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "DIABAGE2",
    "SEX",
    "MARITAL",
    "EDUCA",
    "RENTHOM1",
    "NUMHHOL2",
    "NUMPHON2",
    "CPDEMO1",
    "VETERAN3",
    "EMPLOY1",
    "CHILDREN",
    "INCOME2",
    "INTERNET",
    "WEIGHT2",
    "HEIGHT3",
    "PREGNANT",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "SMOKE100",
    "SMOKDAY2",
    "STOPSMK2",
    "LASTSMK2",
    "USENOW3",
    "ALCDAY5",
    "AVEDRNK2",
    "DRNK3GE5",
    "MAXDRNKS",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXERANY2",
    "EXRACT11",
    "EXEROFT1",
    "EXERHMM1",
    "EXRACT21",
    "EXEROFT2",
    "EXERHMM2",
    "STRENGTH",
    "LMTJOIN3",
    "ARTHDIS2",
    "ARTHSOCL",
    "JOINPAIN",
    "SEATBELT",
    "FLUSHOT6",
    "FLSHTMY2",
    "IMFVPLAC",
    "PNEUVAC3",
    "HIVTST6",
    "HIVTSTD3",
    "WHRTST10",
    "PDIABTST",
    "PREDIAB1",
    "INSULIN",
    "BLDSUGAR",
    "FEETCHK2",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "EYEEXAM",
    "DIABEYE",
    "DIABEDU",
    "CAREGIV1",
    "CRGVREL1",
    "CRGVLNG1",
    "CRGVHRS1",
    "CRGVPRB1",
    "CRGVPERS",
    "CRGVHOUS",
    "CRGVMST2",
    "CRGVEXPT",
    "VIDFCLT2",
    "VIREDIF3",
    "VIPRFVS2",
    "VINOCRE2",
    "VIEYEXM2",
    "VIINSUR2",
    "VICTRCT4",
    "VIGLUMA2",
    "VIMACDG2",
    "CIMEMLOS",
    "CDHOUSE",
    "CDASSIST",
    "CDHELP",
    "CDSOCIAL",
    "CDDISCUS",
    "WTCHSALT",
    "LONGWTCH",
    "DRADVISE",
    "ASTHMAGE",
    "ASATTACK",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ASACTLIM",
    "ASYMPTOM",
    "ASNOSLEP",
    "ASTHMED3",
    "ASINHALR",
    "HAREHAB1",
    "STREHAB1",
    "CVDASPRN",
    "ASPUNSAF",
    "RLIVPAIN",
    "RDUCHART",
    "RDUCSTRK",
    "ARTTODAY",
    "ARTHWGT",
    "ARTHEXER",
    "ARTHEDU",
    "TETANUS",
    "HPVADVC2",
    "HPVADSHT",
    "SHINGLE2",
    "HADMAM",
    "HOWLONG",
    "HADPAP2",
    "LASTPAP2",
    "HPVTEST",
    "HPLSTTST",
    "HADHYST2",
    "PROFEXAM",
    "LENGEXAM",
    "BLDSTOOL",
    "LSTBLDS3",
    "HADSIGM3",
    "HADSGCO1",
    "LASTSIG3",
    "PCPSAAD2",
    "PCPSADI1",
    "PCPSARE1",
    "PSATEST1",
    "PSATIME",
    "PCPSARS1",
    "PCPSADE1",
    "PCDMDECN",
    "SCNTMNY1",
    "SCNTMEL1",
    "SCNTPAID",
    "SCNTWRK1",
    "SCNTLPAD",
    "SCNTLWK1",
    "SXORIENT",
    "TRNSGNDR",
    "RCSGENDR",
    "RCSRLTN2",
    "CASTHDX2",
    "CASTHNO2",
    "EMTSUPRT",
    "LSATISFY",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "MISTMNT",
    "ADANXEV",
    "QSTVER",
    "QSTLANG",
    "MSCODE",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
    "_RFHLTH",
    "_HCVU651",
    "_RFHYPE5",
    "_CHOLCHK",
    "_RFCHOL",
    "_LTASTH1",
    "_CASTHM1",
    "_ASTHMS1",
    "_DRDXAR1",
    "_PRACE1",
    "_MRACE1",
    "_HISPANC",
    "_RACE",
    "_RACEG21",
    "_RACEGR3",
    "_RACE_G1",
    "_AGEG5YR",
    "_AGE65YR",
    "_AGE80",
    "_AGE_G",
    "HTIN4",
    "HTM4",
    "WTKG3",
    "_BMI5",
    "_BMI5CAT",
    "_RFBMI5",
    "_CHLDCNT",
    "_EDUCAG",
    "_INCOMG",
    "_SMOKER3",
    "_RFSMOK3",
    "DRNKANY5",
    "DROCDY3_",
    "_RFBING5",
    "_DRNKWEK",
    "_RFDRHV5",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
    "_MISFRTN",
    "_MISVEGN",
    "_FRTRESP",
    "_VEGRESP",
    "_FRUTSUM",
    "_VEGESUM",
    "_FRTLT1",
    "_VEGLT1",
    "_FRT16",
    "_VEG23",
    "_FRUITEX",
    "_VEGETEX",
    "_TOTINDA",
    "METVL11_",
    "METVL21_",
    "MAXVO2_",
    "FC60_",
    "ACTIN11_",
    "ACTIN21_",
    "PADUR1_",
    "PADUR2_",
    "PAFREQ1_",
    "PAFREQ2_",
    "_MINAC11",
    "_MINAC21",
    "STRFREQ_",
    "PAMISS1_",
    "PAMIN11_",
    "PAMIN21_",
    "PA1MIN_",
    "PAVIG11_",
    "PAVIG21_",
    "PA1VIGM_",
    "_PACAT1",
    "_PAINDX1",
    "_PA150R2",
    "_PA300R2",
    "_PA30021",
    "_PASTRNG",
    "_PAREC1",
    "_PASTAE1",
    "_LMTACT1",
    "_LMTWRK1",
    "_LMTSCL1",
    "_RFSEAT2",
    "_RFSEAT3",
    "_FLSHOT6",
    "_PNEUMO2",
    "_AIDTST3",
]


# check version number"""


# Keep only the meaningful columns
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

'''
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


over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [("o", over), ("u", under)]
pipeline = Pipeline(steps=steps)
x, y = pipeline.fit_resample(x, y)

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

gb_classifier = GradientBoostingClassifier(
    n_estimators=100,
    subsample=0.1,
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_depth=4,
    learning_rate=0.1,
)
gb_classifier.fit(x, y)
prediction = gb_classifier.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
create_csv_submission(test_ids, prediction, "y_predGBC.csv")  # F1:    | Acc:"""


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
