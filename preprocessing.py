import numpy as np
from helpers import *
from implementations import *
from random import randint

### Over and Under sampling


class RandomUnderSampler:
    def __init__(self, sampling_strategy="auto", random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class = unique_classes[np.argmin(class_counts)]
        maj_class = unique_classes[np.argmax(class_counts)]

        min_count = class_counts[np.argmin(class_counts)]
        maj_count = class_counts[np.argmax(class_counts)]

        if self.sampling_strategy == "auto":
            ratio = float(maj_count) / min_count
        else:
            ratio = float(self.sampling_strategy)

        if ratio >= 1.0:
            return X, y  # No need to undersample

        num_samples_to_keep = int(maj_count * ratio)

        maj_indices = np.where(y == maj_class)[0]
        min_indices = np.where(y == min_class)[0]

        if num_samples_to_keep >= len(maj_indices):
            return X, y  # No need to undersample

        selected_maj_indices = np.random.choice(
            maj_indices, size=num_samples_to_keep, replace=False
        )
        combined_indices = np.concatenate([selected_maj_indices, min_indices])

        X_resampled = X[combined_indices]
        y_resampled = y[combined_indices]

        return X_resampled, y_resampled


class RandomOverSampler:
    def __init__(self, sampling_strategy=1.0, random_state=None):
        """
        Initialize the RandomOverSampler.

        Parameters:
        sampling_strategy : float, optional (default=1.0)
            The desired ratio of the number of samples in the minority class over
            the number of samples in the majority class after oversampling. For
            example, if you want a 1:2 ratio, set sampling_strategy=0.5.

        random_state : int or None, optional (default=None)
            Seed for random number generation to ensure reproducibility.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_generate(self, X, y):
        """
        Random oversample the minority class in the dataset to achieve a specific
        sampling strategy (ratio).

        Parameters:
        X : numpy array or array-like
            Feature matrix of shape (n_samples, n_features).

        y : numpy array or array-like
            Target labels of shape (n_samples,).

        Returns:
        X_resampled : numpy array
            Oversampled feature matrix.

        y_resampled : numpy array
            Oversampled target labels.
        """
        # Check if sampling_strategy is a float between 0 and 1
        if not (0 < self.sampling_strategy <= 1.0):
            raise ValueError("sampling_strategy must be a float between 0 and 1.")

        # Identify minority and majority classes
        unique_classes, class_counts = np.unique(y, return_counts=True)
        minority_class = unique_classes[np.argmin(class_counts)]
        majority_class = unique_classes[np.argmax(class_counts)]

        # Determine the number of samples needed to achieve the desired ratio
        majority_samples = class_counts[unique_classes == majority_class][0]
        minority_samples = class_counts[unique_classes == minority_class][0]
        target_minority_samples = int(majority_samples * self.sampling_strategy)

        # Randomly oversample the minority class
        random_state = np.random.default_rng(seed=self.random_state)
        minority_indices = np.where(y == minority_class)[0]
        oversampled_indices = random_state.choice(
            minority_indices, size=target_minority_samples, replace=True
        )

        X_minority = X[oversampled_indices]
        y_minority = y[oversampled_indices]

        # Combine oversampled minority class with majority class
        X_resampled = np.vstack((X, X_minority))
        y_resampled = np.hstack((y, y_minority))

        return X_resampled, y_resampled


### ANOVA


def square(mat):
    m = np.mean(mat, axis=0)


def anova_f(x, y, strat, k=20, use_autosave=True):
    if use_autosave:
        print(
            "WARNING : the autosave feature does not distinguish parameters at more than 1e-3 precision"
        )
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
                categories = np.unique(x[:, feature])
                means_categ = {c: np.mean(y[x[:, feature] == c]) for c in categories}
                ss_c = np.sum(
                    [
                        (means_categ[c] - overall_mean) ** 2 * sum(x[:, feature] == c)
                        for c in categories
                    ]
                )
                ss.append(ss_c)
                df_c = len(categories) - 1
                df.append(df_c)
                ms_c = ss_c / df_c
                ms.append(ms_c)
            ss_error = np.sum((y - overall_mean) ** 2 - np.sum(ss))
            df_error = len(y) - np.sum(df) + 1
            ms_error = ss_error / df_error
            f_stat = []
            for feature in range(nb_feature):
                f_stat.append(ms[feature] / ms_error)

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
            categories = np.unique(x[:, feature])
            means_categ = {c: np.mean(y[x[:, feature] == c]) for c in categories}
            ss_c = np.sum(
                [
                    (means_categ[c] - overall_mean) ** 2 * sum(x[:, feature] == c)
                    for c in categories
                ]
            )
            ss.append(ss_c)
            df_c = len(categories) - 1
            df.append(df_c)
            ms_c = ss_c / df_c
            ms.append(ms_c)
        ss_error = np.sum((y - overall_mean) ** 2 - np.sum(ss))
        df_error = len(y) - np.sum(df) + 1
        ms_error = ss_error / df_error
        f_stat = []
        for feature in range(nb_feature):
            f_stat.append(ms[feature] / ms_error)
    top_k_indices = np.argpartition(f_stat, -k)[-k:]

    # Create a boolean array of length k with True at the indices of the k largest values
    k_best = np.zeros(len(f_stat), dtype=bool)
    k_best[top_k_indices] = True

    return k_best


### Preprocessing

def transform(X, f):
    l = []
    for i in range(len(f)):
        if f[i]:
            l.append(i)
    print(l)
    return np.transpose(np.transpose(X)[l])


def preprocessing(
    X_train,
    X_test,
    Y_train,
    Kselected,
    sampling_strat1=0.105,
    sampling_strat2=0.5,
):
    imp = SimpleImputer()
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp = imp.fit(X_test)
    X_test = imp.transform(X_test)

    X_t = np.delete(X_train, [9, 11, 12, 18, 19, 22], 1)
    X_t2 = np.delete(X_test, [9, 11, 12, 18, 19, 22], 1)

    print(">>> Before resample: \n --------------------------------------")
    print(
        "ytrain -1: "
        + str(np.count_nonzero(Y_train == -1))
        + "   |  ytrain 1: "
        + str(np.count_nonzero(Y_train == 1))
        + "\n-------------------------------------\n"
    )
    ros = RandomOverSampler(sampling_strategy=sampling_strat1)
    X_balanced, Y_balanced = ros.fit_generate(X_t, Y_train)
    rus = RandomUnderSampler(sampling_strategy=sampling_strat2)
    X_t, Y_t = rus.fit_resample(X_balanced, Y_balanced)

    print(">>> After resample: \n --------------------------------------")
    print(
        "ytrain -1: "
        + str(np.count_nonzero(Y_t == -1))
        + "   |  ytrain 1: "
        + str(np.count_nonzero(Y_t == 1))
        + "\n-------------------------------------\n"
    )
    print("Over/Under sampling done")

    strat = str(int(sampling_strat1 * 1000)) + "_" + str(int(sampling_strat2 * 1000))
    fs = anova_f(X_t, Y_t, strat, k=Kselected, use_autosave=True)
    X_t = transform(X_t, fs)
    X_t2 = transform(X_t2, fs)
    # f = fs.get_support(1)

    print("K Best done")

    return X_t, X_t2, Y_t
