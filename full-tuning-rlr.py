from preprocessing import *
from utils import *
from helpers import *
from score import *

"""
In this script, we will optimize the parameters for regularized logistic regression.
We will procede in two steps :
- Step 1 : simple grid search
- Step 2 : local search, with a starting point given by step 1
Step 2 is explain in depth just before its code


IN THIS CODE, X_TEST IS THE 20% OF DATA WE KEEP FOR VALIDATION AND X_ASSESSMENT IS THE REAL TEST (SUBMISSION) DATA
"""

fscores_k = np.loadtxt("f_scores_after_strat105_500.csv")

def partialPreprocessing(X_train, X_test, Y_train, sampling_strat1=0.105, sampling_strat2=0.5):
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
  idx = np.argpartition(fscores, k)
  return x_train[:,idx[:k]]


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

"""
## LOCAL SEARCH ###################################################################
# Improve this fine-tuning with kangaroo-search
# Now we will only try to maximize f-score

"""
"""
The idea is to search randomly around the position of grid search optimum.
At each iteration (maximum number of iterations : max_jumps) :
    - Create a number of new points : for each point in points, add to points [lambda +-0  pas_lambda, gamma +-0 pas_gamma]
    - Here, pas_lambda = saut[0]/ratio et pas_gamma = saut[1]/ratio
    - Calculate f_score for each of these new points and old points (less efficient but clearer)
    - If the best fscore is the same as the old one, multiply ratio by 2
    - Keep in points only nb_points (those with best f-scores) 
    - Number of trainings : nb_points*max_jumps*9 (ici le 9 vient du fait que l'on optimise 2 paramètres : 3² = 9)
"""
"""
print("Entering local-search fine-tuning - gamma and lambda only")

k = k_best
max_iter = max_iter_best

X_trk = kbest(X_train, k, fscores_k)
X_tek = kbest(X_test, k, fscores_k)

X_trk = standardize(X_trk)
X_tek = standardize(X_tek)

    
y_train, tx_train = build_model_data(X_trk, y_train)

w = np.zeros(np.shape(tx_train)[1])

points = [[lambda_best, gamma_best]]
saut = [lambda_best/2, gamma_best/2]
ratio = 1
max_jumps = 8
nb_points = 2
best_f = f_score_best



for iter in range(max_jumps): 
    print(f"Local search, iteration {iter+1} out of {max_jumps}")

    # Create all deltas to make new points
    pas_lambda = saut[0]/ratio
    pas_gamma = saut[1]/ratio
    delta_lambda = [pas_lambda, -pas_lambda, 0]
    delta_gamma = [pas_gamma, -pas_gamma, 0]

    # Create new points
    new_points = []
    for p in points:
        new_points.append(p)
        for g in delta_gamma:
            for l in delta_lambda:
                new_points.append([max(0, p[0] + l), max(0, p[1] + g)])
    
    # Compute f_scores for every point
    f_scores = []
    for p in new_points:
        lambda_, gamma = p[0], p[1]
        w_opti, _ = reg_logistic_regression(y_train, tx_train, lambda_, w, max_iter, gamma, )
        y_pred = predict(X_tek, w_opti,proba=True)
        f_scores.append(f1_score(y_test, y_pred))

    # if the f-score did not improve, refine by dividing the ratio
    if best_f==max(f_scores):
        ratio = ratio/2
    
    best_f = max(f_scores)

    # keep only the np_points best points
    points = []
    index = f_scores.index(best_f)
    points.append(new_points[index])

    new_points.pop(index)
    f_scores.pop(index)

    for i in range(1, nb_points):
        index = f_scores.index(max(f_scores))
        points.append(new_points[index])

        new_points.pop(index)
        f_scores.pop(index)

# Because of the way we pop/append, points is ordered by decresing f-score
lambda_ = points[0][0]
gamma = points[0][1]

    
print(f"After local search, new best f-score : {best_f}, for lambda = {lambda_} and gamma = {gamma}")
"""
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

