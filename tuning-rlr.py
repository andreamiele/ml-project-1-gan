from preprocessing import *
from utils import *
from helpers import *
from score import *


# Loading the datas and splitting them

def create_train_test_split(X, y, test_size=0.20, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

X, Y, _, _ = load_csv_data("dataset/")

X_train, X_test, y_train, y_test = create_train_test_split(X, Y)

X_train, X_test, y_train = preprocessing(X_train, X_test, y_train)

# Defining the test parameters

lambdas = [i/100 for i in range(1, 101, 10)]
gammas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1]

w = np.ones(np.shape(X_train)[1])

f_scores = np.zeros((len(lambdas), len(gammas)))
accuracies = np.zeros((len(lambdas), len(gammas)))

for i, lambda_ in enumerate(lambdas):
    for j, gamma in enumerate(gammas):
        w_opti, _ = reg_logistic_regression(y_train, X_train, lambda_, w, gamma)
        y_pred = predict(X_test, w)
        f_scores[i][j] = f1_score(y_test, y_pred)
        accuracies[i][j] = accuracy_score(y_test, y_pred)

# Choosing best lambda, gamma

i_f, j_f = np.argmax(f_scores)
lambda_f, gamma_f = lambdas[i_f], gammas[j_f]
accuracy_f = accuracies[i_f][j_f]
f_score_f = f_scores[i_f][j_f]

i_a, j_a = np.argmax(accuracies)
lambda_a, gamma_a = lambdas[i_a], gammas[j_a]
accuracy_a = accuracies[i_a][j_a]
f_score_a = f_scores[i_a][j_a]

print("Lambda and gamma fine-tuning...")
print(f"If f-score is your goal : best possible f-score is {f_score_f} with lambda = {lambda_f} and gamma = {gamma_f}.")
print(f"In this case, this accuracy is {accuracy_f}")
print(f"If accuracy is your goal : best possible accuracy is {accuracy_a} with lambda = {lambda_a} and gamma = {gamma_a}.")
print(f"In this case, the f-score is {f_score_a}")

# Improve this fine-tuning with kangaroo-search
# Now we will only try to maximize f-score

points = [[lambda_f, gamma_f]]
saut = [0.1, 0.1]
ratio = 1
max_jumps = 10
nb_points = 2
best_f = f_score_f

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

for iter in range(max_jumps): 
    print(f"Local search, iteration {iter} :")
    print(f"Best f-score : {best_f}, and ratio = {ratio}")
    pas_lambda = saut[0]/ratio
    pas_gamma = saut[1]/ratio
    delta_lambda = [pas_lambda, -pas_lambda, 0]
    delta_gamma = [pas_gamma, -pas_gamma, 0]
    new_points = []
    for p in points:
        new_points.append(p)
        for g in delta_gamma:
            for l in delta_lambda:
                new_points.append([p[0] + l, p[1] + g])
    f_scores = []
    for p in new_points:
        lambda_, gamma = p[0], p[1]
        w_opti, _ = reg_logistic_regression(y_train, X_train, lambda_, w, gamma)
        y_pred = predict(X_test, w)
        f_scores.append(f1_score(y_test, y_pred))

    points = []

    if best_f==max(f_scores):
        ratio = ratio/2

    best_f = max(f_scores)

    index = f_scores.index(best_f)
    points.append(new_points[index])

    new_points.pop(index)
    f_scores.pop(index)

    for i in range(1, nb_points):
        index = f_scores.index(max(f_scores))
        points.append(new_points[index])

        new_points.pop(index)
        f_scores.pop(index)

lambda_ = points[0][0]
gamma = points[0][1]

    
print(f"After local search, new best f-score : {best_f}, for lambda = {lambda_} and gamma = {gamma}")


