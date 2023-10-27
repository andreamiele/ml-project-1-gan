import numpy as np


def get_confusion_matrix(y_true, y_pred):
    # Calculate True Negatives (TN)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    # Return the confusion matrix as a 2x2 numpy array
    return np.array([[TN, FP], [FN, TP]])


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


def f1_score(y_true, y_pred):
    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate True Positives, False Positives, and False Negatives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == -1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == -1))

    # Calculate precision and recall
    if true_positives + false_positives == 0:
        print("WARNING : prediction of [-1 ... -1]")
        return 0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    if precision + recall == 0:
        f1 = 0
    else:
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1
