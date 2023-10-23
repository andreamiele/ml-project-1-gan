import numpy as np


def get_confusion_matrix(y_true, y_pred):
    # Calculate True Negatives (TN)
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    # Calculate True Positives (TP)
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    # Calculate False Negatives (FN)
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    # Calculate False Positives (FP)
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    # Return the confusion matrix as a 2x2 numpy array
    return np.array([[TN, FP], [FN, TP]])


def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Input arrays must have the same length."
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


def f1_score(y_true, y_pred):
    TN, FP, FN, TP = get_confusion_matrix(y_true, y_pred).ravel()
    # Calculate precision score
    precision_score, recall_score = TP / (TP + FP), TP / (TP + FN)
    # Calculate f1-score
    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score)
    return f1_score
