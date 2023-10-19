from helpers import *
from implementations import *
from imblearn.over_sampling import SMOTE

y_train = np.genfromtxt(
    "y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0
)
x_train = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)
x_test = np.genfromtxt("x_test_processed.csv", delimiter=" ", skip_header=0)
ids = np.genfromtxt("test_ids.csv", delimiter=",")

"""x_test = x_test.T
for col in x_test:
    avg = 0
    card = 0
    for v in col:
        if not np.isnan(v):
            card += 1
            avg += v
    if card != 0:
        col = np.where(np.isnan(col), avg / card, col)
x_test = x_test.T"""
col_means = np.nanmean(x_test, axis=0)
x_test = np.where(np.isnan(x_test), col_means, x_test)

initial_w = np.zeros(x_train.shape[1])


def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))


def train_logistic_regression(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for epoch in range(num_iterations):
        model = np.dot(X, weights)
        predictions = sigmoid(model)
        gradient = (1 / num_samples) * np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradient
        if (epoch % 100 == 0) or (epoch == num_iterations - 1):
            print(f"Epoch {epoch}/{num_iterations - 1}")

    return weights


def predict(X, weights):
    model = np.dot(X, weights)
    predictions = sigmoid(model)
    return predictions


# Train the logistic regression model
learning_rate = 0.01
num_iterations = 1000
weights = train_logistic_regression(x_train, y_train, learning_rate, num_iterations)

print("x test isnan: " + str(np.count_nonzero(np.isnan(x_test))))
print("y train isnan: " + str(np.count_nonzero(np.isnan(y_train))))
print("x train isnan: " + str(np.count_nonzero(np.isnan(x_train))))


# Make predictions using the trained model
y_pred = predict(x_test, weights)
print("y pred isnan: " + str(np.count_nonzero(np.isnan(y_pred))))
print(np.median(y_pred))
print(y_pred)
y_pred = np.where(y_pred <= 0.5, -1, 1)
print("y pred isnan: " + str(np.count_nonzero(y_pred == -1)))
create_csv_submission(ids, y_pred, "y_pred3.csv")


from sklearn.linear_model import LogisticRegressionCV

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
x, x_test, y, _, test_ids = load_csv_data("dataset/")
print("pred -1: " + str(np.count_nonzero(y == -1)))
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
x_test = x_test[:, indexColumnsToKeep]

col_means = np.nanmean(x_test, axis=0)
x_test = np.where(np.isnan(x_test), col_means, x_test)

oversample = SMOTE()
x, y = oversample.fit_resample(x, y)
logreg = LogisticRegressionCV(max_iter=1000)
logreg.fit(x, y)
prediction = logreg.predict(x_test)
print("pred -1: " + str(np.count_nonzero(prediction == -1)))
print(prediction)

create_csv_submission(test_ids, prediction, "y_pred4.csv")
