# src/event_detection/evaluation.py
# Threshold tuning and scoring utilities for binary classifiers.

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def get_scores(model, X) -> tuple[np.ndarray | None, str | None]:
    """
    Extract continuous scores from a fitted sklearn Pipeline.

    Tries ``predict_proba`` first (calibrated probabilities), then falls
    back to ``decision_function`` (signed margin).  Returns ``(None, None)``
    if neither is available.

    Parameters
    ----------
    model : sklearn Pipeline
        Fitted pipeline whose last step is the classifier.
    X : array-like
        Feature matrix passed to the pipeline.

    Returns
    -------
    scores : np.ndarray | None
        1-D array of scores for the positive class, or None.
    score_type : str | None
        ``'proba'``, ``'decision'``, or None.
    """
    clf = model.named_steps["classifier"]

    if hasattr(clf, "predict_proba"):
        return model.predict_proba(X)[:, 1], "proba"
    if hasattr(clf, "decision_function"):
        return model.decision_function(X), "decision"

    return None, None


def tune_threshold_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float | None, float | None]:
    """
    Find the decision threshold that maximises F1 on a validation set.

    Uses the full precision-recall curve so that the search is exhaustive
    over all score values without requiring a manual grid.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0 / 1).
    scores : np.ndarray
        Continuous scores for the positive class.

    Returns
    -------
    best_threshold : float | None
        Score value at which F1 is maximised, or None if the curve is
        degenerate (e.g. all predictions identical).
    best_f1 : float | None
        Corresponding F1 value, or None.

    Notes
    -----
    ``precision_recall_curve`` returns arrays of length n_thresholds + 1
    for precision and recall, and n_thresholds for thresholds.  The first
    element of each metric array corresponds to the trivial classifier that
    predicts all positives, so we drop it to align indices with thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if len(thresholds) == 0:
        return None, None

    # Drop the extra leading element so all three arrays are co-indexed
    precision = precision[:-1]
    recall    = recall[:-1]

    denom     = precision + recall
    f1_scores = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def predict_with_threshold(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert continuous scores to binary predictions using a custom threshold.

    Parameters
    ----------
    scores : np.ndarray
        Continuous scores for the positive class.
    threshold : float
        Decision boundary; samples with score >= threshold are labelled 1.

    Returns
    -------
    np.ndarray
        Integer array of 0 / 1 predictions.
    """
    return (scores >= threshold).astype(int)