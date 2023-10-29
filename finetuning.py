from helpers import *
from implementations import *
from preprocessing import *
from utils import *
from score import *

max_iters = 1000
gamma = 0.01


_threshold = 0.5
_kbest = 150
_sampling_strat1 = 0.105
_sampling_strat2 = 0.5
_degree = 3


learn_rate = [1e-2]  # Learning Rates
d_ = [3]  # Max Rates
k_ = [153]  # K for Select k best
s1_ = [0.001 * i for i in range(100, 250, 1)]
s2_ = [0.1 * i for i in range(5, 9, 1)]
r_ = [0.0001 * i for i in range(8500, 9000, 1)]  # Threshold in quantile
print(r_)


def main(training=False):
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
    test_ids = test_ids.astype(dtype=int)
    f1_ = 0
    acc_ = 0
    params = [0, 0, 0, 0, 0, 0]
    if training:
        x_train, x_test, y_train, y_test = split_data(y_train, x_train, 0.75)
        print(x_train.shape[0])
        for s1 in s1_:
            for s2 in s2_:
                for k in k_:

                    x_train2, x_test2, yb_train2 = preprocessing(
                        x_train, x_test, y_train, k, s1, s2
                    )

                    # tx = build_poly(x_train2, degree=_degree)
                    # tx_test = build_poly(x_test2, degree=_degree)
                    # w, loss = ridge_regression(yb_train, tx, lambda_=_lambda)
                    # y_pred = predict(tx_test, w, threshold=_threshold, proba=False)

                    x_train2 = standardize(x_train2)
                    x_test2 = standardize(x_test2)
                    # yb_train2, tx = build_model_data(x_train2, yb_train2)
                    # tx = build_poly(x_train2, degree=_degree)
                    # initial_w = np.zeros(tx.shape[1])

                    # tx_test = build_poly(x_test2, degree=_degree)
                    for d in d_:
                        tx = build_poly(x_train2, degree=d)
                        tx_test = build_poly(x_test2, degree=d)
                        for l in learn_rate:

                            w, loss = ridge_regression(yb_train2, tx, lambda_=l)
                            # w,loss = mean_squared_error_sgd(yb_train2, tx, initial_w, max_iters, gamma)

                            if training:
                                for r in r_:
                                    print(
                                        f"===================\nTesting for {k}/{d}/{l}/{r}/{s1}/{s2}"
                                    )
                                    y_pred = predict(
                                        tx_test, w, threshold=r, proba=False, poly=True
                                    )
                                    # y_pred = predict(x_test, w, threshold=r, proba=False, poly=False)
                                    print(r)
                                    f1 = f1_score(y_test, y_pred)
                                    if f1 > f1_:
                                        f1_ = f1
                                        params = [k, d, l, r, s1, s2]
                                    accuracy = accuracy_score(y_test, y_pred)
                                    print("f1: " + str(f1))
                                    print("acc: " + str(accuracy))
                                    print(
                                        "------------------------\nmax f1: " + str(f1_)
                                    )
                                    print("max params: " + str(params))
    else:
        x_train2, x_test2, yb_train2 = preprocessing(
            x_train, x_test, y_train, 35, _sampling_strat1, _sampling_strat2
        )
        x_train2 = standardize(x_train2)
        x_test2 = standardize(x_test2)
        yb_train2, tx = build_model_data(x_train2, yb_train2)
        initial_w = np.zeros(tx.shape[1])
        w, loss = mean_squared_error_sgd(yb_train2, tx, initial_w, 1000, 0.001)
        y_pred = predict(x_test2, w, threshold=0.84, proba=False, poly=False)
        create_csv_submission(test_ids, y_pred, "resultatSGD.csv")
        print("Training done and exported")
    print("Finished")
    print(f1_)
    print(params)
    return 0


main(True)
