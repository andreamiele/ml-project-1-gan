import numpy as np

def square(mat):
    m = np.mean(mat, axis = 0)

def anova_f(x, y, strat, k = 20, use_autosave = True):
    if use_autosave:
        print("WARNING : the autosave feature does not distinguish parameters at more than 1e-3 precision")
        try:
            f_stat = list(np.loadtxt("f_scores_after_strat" + strat + ".csv"))
        except:
            print("Creating the F-score file - this might take a while")
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
            
            np.savetxt("f_scores_after_strat" + strat + ".csv", f_stat, delimiter=",")
    else:
        print("Creating the F-score file - this might take a while")
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
