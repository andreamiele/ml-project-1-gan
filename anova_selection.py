import numpy as np

def square(mat):
    m = np.mean(mat, axis = 0)

def anova_f(x, y, k = 20):
    nb_data, nb_feature = np.shape(x)
    overall_mean = np.mean(y)
    ss = []
    df = []
    ms = []
    for feature in range(nb_feature):
        print(feature)
        categories = np.unique(x[:,feature])
        means_categ = {c: np.mean(y[x[:,feature] == c]) for c in categories}
        ss_c = np.sum([(means_categ[c] - overall_mean) ** 2 * sum(x[:,feature] == c) for c in categories])
        ss.append(ss_c)
        df_c = len(categories) - 1
        df.append(df_c)
        ms_c = ss_c/df_c
        ms.append(ms_c)
    ss_error = np.sum((y - overall_mean) ** 2 - np.sum(ss))
    df_error = len(y) - np.sum(df) + 1
    ms_error = ss_error/df_error
    f_stat = []
    for feature in range(nb_feature):
        f_stat.append(ms[feature]/ms_error)
    
    top_k_indices = np.argpartition(f_stat, -k)[-k:]

    # Create a boolean array of length k with True at the indices of the k largest values
    k_best = np.zeros(len(f_stat), dtype=bool)
    k_best[top_k_indices] = True

    return k_best
