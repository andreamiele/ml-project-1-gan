# Import helper functions
from helpers import *
from implementations import *
from preprocessing import *
from utilities import *

"""
def logreg_grad_sgd(y, tx, w):
    index = np.random.randint(0, y.shape[0] - 1)
    pred = sigmoid(tx[index].dot(w))
    return tx[index] * (pred - y[index])


def logistic_regression_sgd(y, tx, initial_w, hyperparameters):
    max_iters = hyperparameters[0]
    gamma = hyperparameters[1]
    w = initial_w
    for n_iter in range(max_iters):
        g = logreg_grad_sgd(y, tx, w)

        wold = w
        w = w - gamma * g
        if np.linalg.norm(w - wold) == 0:
            break
    return w


def reg_logistic_regression_CV_friendly(y, tx, w, hyperparameters):
    return reg_logistic_regression(
        y, tx, hyperparameters[0], w, hyperparameters[1], hyperparameters[2]
    )


x_test = x_test.T
for col in x_test:
    avg = 0
    card = 0
    for y in col:
        if not np.isnan(y):
            card += 1
            avg += y
    if card != col.shape[0]:
        col[np.isnan(col)] = avg / card
x_test = x_test.T
tmp = np.ones((x_train.shape[0], x_train.shape[1] + 1))
tmp[:, 1:] = x_train
x_train = tmp

y_train[y_train == -1] = 0

tmp = np.ones((x_test.shape[0], x_test.shape[1] + 1))
tmp[:, 1:] = x_test
x_test = tmp
w = np.zeros(x_test.shape[1])

gammas = np.arange(200) * 0.005
max_iters = np.array([1000])
hyperparameters = []
for mi in max_iters:
    for g in gammas:
        hyperparameters.append([mi, g])

# best_hyperparameters, _ = cross_validation(x_train, y_train, w, logistic_regression_sgd, logreg_loss, 4, hyperparameters)

# gamma = best_hyperparameters[1]
# max_iters = best_hyperparameters[0]

w, _ = reg_logistic_regression(y_train, x_train, 0.1, w, 10000, 0.1)

print(w.shape, x_test.shape)
pred = sigmoid(x_test.dot(w))
print(np.mean(pred), np.max(pred), np.min(pred))
mean = np.mean(pred)
pred[pred < mean] = -1
pred[pred >= mean] = 1
count = 0
for a in pred:
    if a == -1:
        count += 1
print(count, pred.shape[0])
create_csv_submission(ids, pred, "submit.csv")
"""

_kbest = 150
_sampling_strat1 = 0.105
_sampling_strat2 = 0.5
_degree = 3
_lambda = 10e-2
_threshold = 0.87


def main(training=False):
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
    test_ids = test_ids.astype(dtype=int)

    if training:
        x_train, x_test, y_train, y_test = create_train_test_split(x_train, y_train)

    x_train, x_test, yb_train = preprocessing(
        x_train, x_test, y_train, _kbest, _sampling_strat1, _sampling_strat2
    )

    x_train2 = standardize(x_train)
    x_test2 = standardize(x_test)

    tx = build_poly(x_train2, degree=_degree)
    tx_test = build_poly(x_test2, degree=_degree)
    w, loss = ridge_regression(yb_train, tx, lambda_=_lambda)

    y_pred = predict(tx_test, w, threshold=_threshold, proba=False)

    if training:
        f1 = f1_score(Y_test, y_pred)
        accuracy = accuracy_score(Y_test, y_pred)
        print("f1: " + str(f1))
        print("acc: " + str(accuracy))
    else:
        create_csv_submission(test_ids, y_pred, "resultat.csv")
    print("Finished")
    return 0


### Run main function
main()
