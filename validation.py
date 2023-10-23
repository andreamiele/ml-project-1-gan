from imp import *
from helpers import *
from run_fonctions import *


def build_k_indices(y, k_fold, seed):
    # Build k indices for k-fold cross validation.#
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


############# GRADIENT DESCENT ################
def cross_validation_gd(y, x, k_indices, k, gamma, max_iters):
    # Perform the k_th step of cross validation on gradient descent. #

    y_te = y[k_indices[k, :]]
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    y_tr, tx_tr = build_model_data(x_tr, y_tr)
    y_te, tx_te = build_model_data(x_te, y_te)
    initial_w = np.zeros(tx_tr.shape[1])
    w, loss = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
    y_pred = predict_labels(w, tx_te)
    acc = float(np.sum(y_te == y_pred)) / len(y_te)

    return acc


def gradientdescent_gamma(y, x):
    # Plots and prints the accuracy for a certain value range of gammas, #
    # for gradient descent                                               #
    seed = 1
    k_fold = 4
    max_iters = 300
    gammas = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data

    accs = []
    stds = []

    for gamma in gammas:
        acc_temp = []
        for k in range(k_fold):
            acc = cross_validation_gd(y, x, k_indices, k, gamma, max_iters)
            acc_temp.append(acc)
        accs.append(np.mean(acc_temp))
        stds.append(np.std(acc_temp))

        print(gamma, ": Acc = ", np.mean(acc_temp), ", std = ", np.std(acc_temp))

    gradientdescent_gamma_visualization(gammas, accs, stds)


############# STOCHASTIC GRADIENT DESCENT ################


def cross_validation_sgd(y, x, k_indices, k, gamma, max_iters):
    # Perform the k_th step of cross validation on gradient descent. #

    y_te = y[k_indices[k, :]]
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    y_tr, tx_tr = build_model_data(x_tr, y_tr)
    y_te, tx_te = build_model_data(x_te, y_te)
    initial_w = np.zeros(tx_tr.shape[1])
    w, loss = least_squares_SGD(y_tr, tx_tr, initial_w, max_iters, gamma)
    y_pred = predict_labels(w, tx_te)
    acc = float(np.sum(y_te == y_pred)) / len(y_te)

    return acc


def stochastic_gradientdescent_gamma(y, x):
    # Plots and prints the accuracy for a certain value range of gammas,   #
    # for gradient descent                                                 #
    seed = 1
    k_fold = 4
    max_iters = 100
    gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data

    accs = []
    stds = []

    for gamma in gammas:
        acc_temp = []
        for k in range(k_fold):
            acc = cross_validation_sgd(y, x, k_indices, k, gamma, max_iters)
            acc_temp.append(acc)
        accs.append(np.mean(acc_temp))
        stds.append(np.std(acc_temp))

        print(gamma, ": Acc = ", np.mean(acc_temp), ", std = ", np.std(acc_temp))

    gradientdescent_gamma_visualization(gammas, accs, stds)


###################### LEAST SQUARES ######################


def cross_validation_ls(y, x, k_indices, k, degree):
    # Perform the k_th step of cross validation on least squares    #
    y_te = y[k_indices[k, :]]
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w, loss = least_squares(y_tr, tx_tr)
    y_pred = predict_labels(w, tx_te)
    acc = float(np.sum(y_te == y_pred)) / len(y_te)

    return acc


def leastsquares_degree(y, x):
    # Plots and prints the accuracy for a certain value range of     #
    # degrees for the polynomial kernel,for least squares           #
    seed = 1
    k_fold = 4
    degrees = range(1, 14 + 1)

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data #

    accs = []
    stds = []

    for degree in degrees:
        acc_temp = []
        for k in range(k_fold):
            acc = cross_validation_ls(y, x, k_indices, k, degree)
            acc_temp.append(acc)
        accs.append(np.mean(acc_temp))
        stds.append(np.std(acc_temp))
        print(degree, ": Acc = ", np.mean(acc_temp), ", std = ", np.std(acc_temp))

    ls_degree_visualization(degrees, accs, stds)


####################### RIDGE REGRESSION #########################


def cross_validation_rr(y, x, k_indices, k, lambda_, degree):
    # Perform the k_th step of cross validation on ridge regression #

    y_te = y[k_indices[k, :]]
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w, loss = ridge_regression(y_tr, tx_tr, lambda_)
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))

    y_pred = predict_labels(w, tx_te)
    acc = float(np.sum(y_te == y_pred)) / len(y_te)

    return loss_tr, loss_te, acc


def ridgeregression_lambda(y, x):
    # Plots and prints the accuracy for a certain value range for lambdas, #
    # for ridge regression                                                 #
    seed = 1
    degree = 10
    k_fold = 4
    lambdas = np.logspace(-5, 0, 6)
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data        #
    rmse_tr = []
    rmse_te = []
    accs = []
    stds = []

    for lambda_ in lambdas:
        rmse_tr_temp = []
        rmse_te_temp = []
        acc_temp = []
        for k in range(k_fold):
            loss_tr, loss_te, acc = cross_validation_rr(
                y, x, k_indices, k, lambda_, degree
            )
            rmse_tr_temp.append(loss_tr)
            rmse_te_temp.append(loss_te)
            acc_temp.append(acc)
        rmse_tr.append(np.mean(rmse_tr_temp))
        rmse_te.append(np.mean(rmse_te_temp))
        accs.append(np.mean(acc_temp))
        stds.append(np.std(acc_temp))

        print(
            degree,
            ", ",
            lambda_,
            ": acc = ",
            np.mean(acc_temp),
            ", std = ",
            np.std(acc_temp),
        )

    rr_lambda_visualization(degree, lambdas, accs, stds)


def ridgeregression_degree_lambda(y, x):
    # Plots and prints the accuracy for a certain value range of      #
    # degrees for the polynomial kernel, and value range for lambdas, #
    # for ridge regression                                            #
    seed = 1
    k_fold = 4
    lambdas = [
        0.0001,
        0.001,
    ]  # [0.000005,0.00001,0.00005,0.0001,0.0005,0.001, 0.005, 0.01]
    degrees = range(9, 11)  # range(4,13)

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    accvectors = []
    stds = []

    for d_ind, lambda_ in enumerate(lambdas):
        # define lists to store the loss of training data and test data
        rmse_tr = []
        rmse_te = []
        acc = []
        std = []
        # cross validation
        for ind, degree in enumerate(degrees):
            loss_tr = []
            loss_te = []
            acc_1 = []
            for k in range(k_fold):
                temp_loss_tr, temp_loss_te, temp_acc = cross_validation_rr(
                    y, x, k_indices, k, lambda_, degree
                )
                loss_tr.append(np.sqrt(2 * temp_loss_tr))
                loss_te.append(np.sqrt(2 * temp_loss_te))
                acc_1.append(temp_acc)
            rmse_tr.append(np.mean(loss_tr))
            rmse_te.append(np.mean(loss_te))
            acc.append(np.mean(acc_1))
            std.append(10 * np.std(acc_1))

            print(
                lambda_,
                ", ",
                degree,
                ": acc = ",
                np.mean(acc_1),
                ", std = ",
                np.std(acc_1),
            )

        if d_ind == 0:
            accvectors = acc
            stds = std
        else:
            accvectors = np.vstack((accvectors, acc))
            stds = np.vstack((stds, std))

    rr_degree_lambda_visualization(degrees, lambdas, accvectors, stds)


############################ LOGISTIC REGRESSION #################################


def cross_validation_lr(y, x, k_indices, k, max_iters, gamma, degree):
    # Returns the loss of ridge regression.     #

    y_te = y[k_indices[k, :]]
    y_te = np.expand_dims(y_te, axis=1)
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    y_tr = np.expand_dims(y_tr, axis=1)
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    initial_w = np.zeros((tx_tr.shape[1], 1))

    w, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
    y_pred = predict_labels(w, tx_te)
    acc = float(np.sum(y_te == y_pred)) / len(y_te)

    return acc


def logregression_gamma(y, x):
    print("start")
    seed = 1
    max_iters = 100
    degree = 1
    k_fold = 4
    gammas = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    accs = []
    for gamma in gammas:
        acc_temp = []
        for k in range(k_fold):
            acc = cross_validation_lr(y, x, k_indices, k, max_iters, gamma, degree)
            acc_temp.append(acc)
        accs.append(np.mean(acc_temp))
        print(gamma, " accuracy = ", np.mean(acc_temp), "std = ", np.std(acc_temp))
    cross_validation_visualization_lr(gammas, accs)


def logregression_gamma_degree(y, x):
    print("start")
    seed = 1
    max_iters = 10
    # degree = 1
    degrees = range(1, 6)
    k_fold = 4
    gammas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.006, 0.01]

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    accs = []
    for d_ind, gamma in enumerate(gammas):
        acc_temp = []
        for ind, degree in enumerate(degrees):
            acc_1 = []
            for k in range(k_fold):
                acc = cross_validation_lr(y, x, k_indices, k, max_iters, gamma, degree)
                acc_1.append(acc)
            acc_temp.append(np.mean(acc_1))
            print(
                gamma,
                degree,
                " accuracy = ",
                np.mean(acc_temp),
                "std = ",
                np.std(acc_temp),
            )
        if d_ind == 0:
            accs = acc_temp
        else:
            accs = np.vstack((accs, acc_temp))
    cross_validation_visualization_lr(gammas, accs)


################### REGURALIZED LOGISTIC REGRESSION ############################


def cross_validation_rlr(y, x, lambda_, k_indices, k, max_iters, gamma):
    # Returns the loss of ridge regression.     #

    y_te = y[k_indices[k, :]]
    y_te = np.expand_dims(y_te, axis=1)
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    y_tr = np.expand_dims(y_tr, axis=1)
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    y_tr, tx_tr = build_model_data(x_tr, y_tr)
    y_te, tx_te = build_model_data(x_te, y_te)

    initial_w = np.zeros((tx_tr.shape[1], 1))

    w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma)
    y_pred = predict_labels(w, tx_te)
    acc = float(np.sum(y_te == y_pred)) / len(y_te)

    return acc


def reglogregression_gamma_lambda(y, x):
    seed = 1
    max_iters = 50

    k_fold = 4
    gammas = [0.001, 0.01, 0.025, 0.1]
    lambdas = [0.0001, 0.001]
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    accs = np.zeros((len(gammas), len(lambdas)))
    for i, gamma in enumerate(gammas):
        for j, lambda_ in enumerate(lambdas):
            acc_temp = []
            for k in range(k_fold):
                acc = cross_validation_rlr(
                    y, x, lambda_, k_indices, k, max_iters, gamma
                )
                acc_temp.append(acc)
            accs[i][j] = np.mean(acc_temp)

        print(
            "gamma: ",
            gamma,
            "lambda: ",
            lambda_,
            " accuracy = ",
            np.mean(acc_temp),
            "std = ",
            np.std(acc_temp),
        )

    cross_validation_visualization_rlr(gammas, lambdas, accs)


############################ STACKING #################################


def cross_validation_stacking(y, x, k_indices, k):
    # Returns the loss of ridge regression.     #

    y_te = y[k_indices[k, :]]
    y_te = np.expand_dims(y_te, axis=1)
    x_te = x[k_indices[k, :]]

    tr_indices = np.delete(k_indices, (k), axis=0)

    y_tr = y[tr_indices].flatten()
    y_tr = np.expand_dims(y_tr, axis=1)
    x_tr = x[tr_indices].reshape(x.shape[0] - x_te.shape[0], x.shape[1])

    y_pred = stacking(y_tr.T, x_tr, y_te.T, x_te)

    acc = float(np.sum(y_te.T == y_pred)) / len(y_te)
    print(acc)
    return acc


def stacking_crossvalidation(y, x):
    seed = 1
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data #
    accs = []
    stds = []
    for k in range(k_fold):
        acc = cross_validation_stacking(y, x, k_indices, k)
        accs.append(acc)
    acc_final = np.mean(accs)
    stds_final = np.std(accs)

    print(" acc = ", acc_final, ", std = ", stds_final)


############################ PLOTS ################################

################# GRADIENT DESCENT PLOTS ##########################
def gradientdescent_gamma_visualization(gammas, accs, stds):
    plt.errorbar(gammas, accs, yerr=stds, marker=".", color="b", label="Accuracy")
    plt.xscale("log")
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Gradient descent cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()


################# LEAST SQUARE PLOTS ########################
def ls_degree_visualization(degs, accs, stds):

    plt.errorbar(degs, accs, yerr=stds, marker=".", color="b", label="lambda = 0.0001")

    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Ridge regression cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()


################# RIDGE REGRESSION PLOTS ########################


def rr_degree_lambda_visualization(degs, lambdas, accs, stds):
    for i in range(len(lambdas)):
        label = "lambda =" + str(lambdas[i])
        plt.errorbar(
            degs, accs[:, i], yerr=stds[:, i], marker=".", color="b", label=label
        )

    plt.xscale("log")
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Ridge regression cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()


def rr_degree_visualization(degs, lambda_, accs, stds):

    plt.errorbar(degs, accs, yerr=stds, marker=".", color="b", label="lambda = 0.0001")

    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("Ridge regression cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()


def rr_lambda_visualization(degree, lambdas, accs, stds):

    plt.errorbar(lambdas, accs, yerr=stds, marker=".", color="b", label="degree = 10")

    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("Ridge regression cross validation")
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()


def cross_validation_visualization(lambds, mse_tr, mse_te):
    # Visualization the curves of mse_tr and mse_te.        #
    plt.semilogx(lambds, mse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, mse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()


################# LOGISTIC REGRESSION PLOTS ########################
def cross_validation_visualization_lr(gammas, acc):
    # Visualization the curves of mse_tr and mse_te.       #
    plt.semilogx(gammas, acc, marker=".", color="b", label="train error")
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.title("cross validation logistic regression")
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()


############ REGULARIZED LOGISTIC REGRESSION PLOTS ##################
def cross_validation_visualization_rlr(gammas, lambdas, accs):

    for i in range(len(lambdas)):
        label = "lambda =" + str(lambdas[i])
        plt.plot(gammas, accs[:, i], marker=".", label=label)

    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.title("cross validation degree accuracy")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()


def cross_validation_visualization_degree(degs, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.plot(degs, mse_tr, marker=".", color="b", label="train error")
    plt.plot(degs, mse_te, marker=".", color="r", label="test error")
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation degree")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_deg")
    plt.show()
