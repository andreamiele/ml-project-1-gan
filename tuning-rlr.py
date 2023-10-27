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


## PREPARATION ########################################################################
# Loading the datas and splitting them

k = 150
max_iter = 100

print("Loading dataset")
X, X_assessment, Y, _, ids = load_csv_data("dataset/")
ids = ids.astype(int)

print("Preprocessing dataset")
X, X_assessment, Y = preprocessing(X, X_assessment, Y, k)

Y, tX = build_model_data(X, Y)

print("Spliting dataset")
X_train, X_test, y_train, y_test = split_data(Y, X, 0.6)

y_train, tx_train = build_model_data(X_train, y_train)

X_train = standardize(X_train)
X_test = standardize(X_test)
X_assessment = standardize(X_assessment)


## GRID SEARCH ########################################################################

# Defining the test parameters

lambdas = np.logspace(-5, 0, num=10)
gammas = np.logspace(-4, 1, num=10)

w = np.zeros(np.shape(tx_train)[1])

f_scores = np.zeros((len(lambdas), len(gammas)))
accuracies = np.zeros((len(lambdas), len(gammas)))

#Simple grid-search optimisation
print("Simple grid-search optimisation")

for i, lambda_ in enumerate(lambdas):
    for j, gamma in enumerate(gammas):
        w_opti, _ = reg_logistic_regression(y_train, tx_train, lambda_, w, max_iter, gamma)
        y_pred = predict(X_test, w_opti, proba=True)
        f = f1_score(y_test, y_pred)
        f_scores[i][j] = f
        accuracies[i][j] = accuracy_score(y_test, y_pred)
        print(f"lambda : {i+1}/{len(lambdas)} ({round(lambda_, 2)}), gamma : {j+1}/{len(gammas)} ({round(gamma, 2)}), f-score = {round(f, 2)}")

# Choosing best lambda, gamma
arg_f = np.argmax(f_scores)
i_f, j_f = arg_f//len(gammas), arg_f%len(gammas)
lambda_f, gamma_f = lambdas[i_f], gammas[j_f]
accuracy_f = accuracies[i_f][j_f]
f_score_f = f_scores[i_f][j_f]

arg_a = np.argmax(accuracies)
i_a, j_a = arg_a//len(gammas), arg_a%len(gammas)
lambda_a, gamma_a = lambdas[i_a], gammas[j_a]
accuracy_a = accuracies[i_a][j_a]
f_score_a = f_scores[i_a][j_a]

print("Lambda and gamma fine-tuning...")
print(f"If f-score is your goal : best possible f-score is {f_score_f} with lambda = {lambda_f} and gamma = {gamma_f}.")
print(f"In this case, this accuracy is {accuracy_f}")
print(f"If accuracy is your goal : best possible accuracy is {accuracy_a} with lambda = {lambda_a} and gamma = {gamma_a}.")
print(f"In this case, the f-score is {f_score_a}")

w_opti, _ = reg_logistic_regression(Y, tX, lambda_f, w, max_iter, gamma_f)
y_pred = predict(X_assessment, w_opti, proba = True)

create_csv_submission(ids, y_pred, "rlr-tuned_grid.csv")


## LOCAL SEARCH ###################################################################
# Improve this fine-tuning with kangaroo-search
# Now we will only try to maximize f-score

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

print("Entering local-search fine-tuning")

points = [[lambda_f, gamma_f]]
saut = [lambda_f/2, gamma_f/2]
ratio = 1
max_jumps = 8
nb_points = 2
best_f = f_score_f



for iter in range(max_jumps): 
    print(f"Local search, iteration {iter} :")
    print(f"Best f-score : {best_f}, and ratio = {ratio}")

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
                new_points.append([p[0] + l, p[1] + g])
    
    # Compute f_scores for every point
    f_scores = []
    for p in new_points:
        lambda_, gamma = p[0], p[1]
        w_opti, _ = reg_logistic_regression(y_train, tx_train, lambda_, w, max_iter, gamma, )
        y_pred = predict(X_test, w_opti,proba=True)
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

## SAVING PREDICITION FOR BEST RESULT #######################################################

w_opti, _ = reg_logistic_regression(Y, tX, lambda_, w, max_iter, gamma)
y_pred = predict(X_assessment, w_opti, proba = True)

create_csv_submission(ids, y_pred, "rlr-tuned-local.csv")

