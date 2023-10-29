# Import helper functions
from helpers import *
from implementations import *
from preprocessing import *
from utils import *

_kbest = 153
_degree = 3
_lambda = 10e-2
_threshold = 0.8751


def main(training=False):
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
    test_ids = test_ids.astype(dtype=int)

    if training:
        x_train, x_test, y_train, y_test = split_data(x_train, y_train)

    x_train, x_test, yb_train = preprocessing(x_train, x_test, y_train, _kbest)

    x_train2 = standardize(x_train)
    x_test2 = standardize(x_test)

    tx = build_poly(x_train2, degree=_degree)
    tx_test = build_poly(x_test2, degree=_degree)
    w, _ = ridge_regression(yb_train, tx, lambda_=_lambda)
    y_pred = predict(tx_test, w, threshold=_threshold, proba=False, poly=True)

    if training:
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("f1: " + str(f1))
        print("acc: " + str(accuracy))
    else:
        create_csv_submission(test_ids, y_pred, "resultat.csv")
    print("Finished")
    return 0


### Run main function
main()
