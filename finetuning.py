from helpers import *
from implementations import *
from preprocessing import *
from utils import *
from score import *

fscores_k = np.loadtxt("f_scores_after_strat105_500.csv")

def partialPreprocessing(X_train, X_test, Y_train, sampling_strat1=0.105, sampling_strat2=0.5):
    #First part of the preprocessing, to not recompute each time the ANOVA 
    imp = SimpleImputer()
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp = imp.fit(X_test)
    X_test = imp.transform(X_test)

    X_t = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
    X_t2 = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)
    ros = RandomOverSampler(sampling_strategy=sampling_strat1)
    X_balanced, Y_balanced = ros.fit_generate(X_t, Y_train)
    rus = RandomUnderSampler(sampling_strategy=sampling_strat2)
    X_t, Y_t = rus.fit_resample(X_balanced, Y_balanced)
    return X_t, Y_t, X_t2

def kbest(x_train, k, fscores):
    #returns the k features of x_train with the best scores precomputed and stored in fscores
    idx = np.argpartition(fscores, k)
    return x_train[:,idx[:k]]

max_iters = 1000
gamma = 0.01


_threshold = 0.5
_kbest = 150
_sampling_strat1 = 0.105
_sampling_strat2 = 0.5
_degree = 3


learn_rate = np.logspace(-5,-1,20)  # Learning Rates
d_ = [2,3,4,5]  # Max Rates
k_ = np.arange(20,200,5)  # K for Select k best
s1_ = [0.001 * i for i in range(100, 250, 1)]
s2_ = [0.1 * i for i in range(5, 9, 1)]
r_ = [0.0001 * i for i in range(8500, 9000, 1)]  # Threshold in quantile


def gradient_descent_finetuning():
    #function used to fine tune the (stochastic) gradient descent methods
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
    test_ids = test_ids.astype(dtype=int)
    f1_ = 0
    params = [0, 0, 0, 0, 0, 0]
    
    x_train, x_test, y_train, y_test = split_data(y_train, x_train, 0.75)

    for k in k_:
    x_train2, yb_train2, x_test2 = partialPreprocessing(
        x_train, x_test, y_train
    )
    x_train2 = kbest(x_train2, k, fscores_k)
    x_test2 = kbest(x_test2, k, fscores_k)
    x_train2 = standardize(x_train2)
    x_test2 = standardize(x_test2)
    
    for d in d_:
        tx = build_poly(x_train2, degree=d)
        tx_test = build_poly(x_test2, degree=d)
        initial_w = np.ones(tx.shape[1])
        for mi in max_iters:
        for l in learn_rate:
            w, _ = mean_squared_error_gd(yb_train2, tx, initial_w, mi, l)
            for r in r_:
            y_pred = predict(
                tx_test, w, threshold=r, proba=False, poly=True
            )
            f1 = f1_score(y_test, y_pred)
            if f1 > f1_:
                f1_ = f1
                params = [k, d, mi, l, r]
                print(params, f1)

    print("Best hyperparameters and associated F1 score: ", params, f1_)
    #params = [312, 1, 1200, 0.02500000000000001, 0.861] 
    #f1_ = 0.40205468457381344
    
    print(params, f1_)           
    x_train2, yb_train2, x_test2 = partialPreprocessing(
    x_train, x_test, y_train
    )
    k = params[0]
    mi = params[2]
    gamma = params[3]
    r = params[4]
    x_train2 = kbest(x_train2, k, fscores_k)
    x_test2 = kbest(x_test2, k, fscores_k)
    x_train2 = standardize(x_train2)
    x_test2 = standardize(x_test2)
    yb_train2, tx = build_model_data(x_train2, yb_train2)
    initial_w = np.zeros(tx.shape[1])
    w, loss = mean_squared_error_gd(yb_train2, tx, initial_w, mi, gamma)
    y_pred = predict(x_test2, w, threshold=r, proba=False, poly=False)
    create_csv_submission(test_ids, y_pred, "gd_fine_tuned.csv")

def ridge_regression_finetuning():
    x_train, x_test, y_train, _, test_ids = load_csv_data("dataset/")
    test_ids = test_ids.astype(dtype=int)
    f1_ = 0
    acc_ = 0
    params = [0, 0, 0, 0, 0, 0]
        
    x_train, x_test, y_train, y_test = split_data(y_train, x_train, 0.75)
    print(x_train.shape[0])
    for s1 in s1_:
        for s2 in s2_:
            for k in k_:

                x_train2, x_test2, yb_train2 = preprocessing(
                    x_train, x_test, y_train, k, s1, s2
                )
                x_train2 = standardize(x_train2)
                x_test2 = standardize(x_test2)
                    
                for d in d_:
                    tx = build_poly(x_train2, degree=d)
                    tx_test = build_poly(x_test2, degree=d)
                    for l in learn_rate:

                        w, loss = ridge_regression(yb_train2, tx, lambda_=l)

                        if training:
                            for r in r_:
                                print(
                                    f"===================\nTesting for {k}/{d}/{l}/{r}/{s1}/{s2}"
                                )
                                y_pred = predict(
                                    tx_test, w, threshold=r, proba=False, poly=True
                                )
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
                                
    print("Finished")
    print(f1_)
    print(params)
    
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
    
def regularized_logreg_finetuning():
    ## PREPARATION ########################################################################


    # Defining the test parameters

    lambdas_n = np.logspace(-5, 0, num=10)
    lambdas_p = np.logspace(-12, 0, num=10)
    gammas = np.logspace(-4, 1, num=10)
    k_list = [50, 150, 300]
    max_iter_list = [20, 100, 500, 1000]
    degre = [1, 2, 3]
    threshold = [0.4, 0.5, 0.6]



    # Loading the datas and splitting them
    print("Loading dataset")
    X, X_assessment, Y, _, ids = load_csv_data("dataset/")
    ids = ids.astype(int)

    print("Preprocessing dataset")
    X, Y, X_assessment = partialPreprocessing(X, X_assessment, Y)

    print("Spliting dataset")
    X_train, X_test, y_train, y_test = split_data(Y, X, 0.8)

    ## GRID SEARCH ########################################################################


    #Simple grid-search optimisation
    print("Simple grid-search optimisation")

    k_best = 0
    max_iter_best = 0
    f_score_best = 0
    lambda_best = 0
    gamma_best = 0
    threshold_best = 0
    degre_best = 0

    compteur = 1

    nb_iters = len(k_list)*len(gammas)*10*len(max_iter_list)*len(threshold)*len(degre)

    for k in k_list:

        X_trk = kbest(X_train, k, fscores_k)
        X_tek = kbest(X_test, k, fscores_k)

        X_trk = standardize(X_trk)
        X_tek = standardize(X_tek)

        for t in threshold:
            for d in degre:
                lambdas = lambdas_p
                pol = True
                X_form = X_tek
                if d==1:
                    y_train, tx_train = build_model_data(X_trk, y_train)
                    pol = False
                    lambdas = lambdas_n
                else:
                    tx_train = build_poly(X_trk, d)
                    X_form = build_poly(X_tek, d)
                w = np.zeros(np.shape(tx_train)[1])

                for max_iter in max_iter_list:
                    for i, lambda_ in enumerate(lambdas):
                        for j, gamma in enumerate(gammas):
                            print(f"Grid optimisation iteration {compteur}/{nb_iters}")
                            w_opti, _ = reg_logistic_regression(y_train, tx_train, lambda_, w, max_iter, gamma)
                            y_pred = predict(X_form, w_opti, proba=True, threshold=t, poly=pol)
                            f = f1_score(y_test, y_pred)
                            if f > f_score_best:
                                lambda_best, gamma_best, max_iter_best, k_best, f_score_best, threshold_best, degre_best = lambda_, gamma, max_iter, k, f, t, d
                            compteur += 1
                
            


    print(f"End of grid optimisation : best parameters are k = {k_best}, max_iter = {max_iter_best}, lambda = {lambda_best} anf gamma = {gamma_best}")
    
    
    
    ## SAVING PREDICITION FOR BEST RESULT #######################################################

    X_trk = kbest(X, k_best, fscores_k)
    X_assessment = kbest(X_assessment, k_best, fscores_k)
    X_assessment = standardize(X_assessment)

    pol = True
    if degre_best==1:
        Y, tx_train = build_model_data(X_trk, Y)
        pol = False
    else:
        tx_train = build_poly(X_trk, degre_best)
        X_assessment = build_poly(X_assessment, degre_best)

    w_opti, _ = reg_logistic_regression(y_train, tx_train, lambda_best, w, max_iter_best, gamma_best)
    y_pred = predict(X_assessment, w_opti, proba = True, poly = pol, threshold=threshold_best)

    create_csv_submission(ids, y_pred, "rlr-full-tuned.csv")