# src/utils/metrics.py
# Classification metrics utilities

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, y_score=None):
    """
    Compute standard binary classification metrics.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        y_score: Prediction scores for ROC-AUC (optional)

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Compute ROC-AUC if scores provided
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except (ValueError, Exception):
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def compute_support(y_true):
    """
    Compute class support (number of samples per class).

    Args:
        y_true: True labels

    Returns:
        Dictionary with support for each class
    """
    y_true = np.asarray(y_true, dtype=int)
    unique, counts = np.unique(y_true, return_counts=True)

    return {int(label): int(count) for label, count in zip(unique, counts)}