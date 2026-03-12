# src/event_detection/model.py
# TF-IDF + shallow classifier pipeline for binary event detection.
#
# Input column: "text"  (expects already-lowercased text_ml)
# Output      : fitted sklearn Pipeline with steps "features" + "classifier"

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Valid model type identifiers
_MODEL_TYPES = {"logreg", "svm", "sgd_logloss", "cnb"}


def build_event_classifier(
    model_type: str,
    seed: int,
    class_weight: str | dict | None,
    max_features: int,
    min_df: int | float,
    max_df: int | float,
    ngram_range: tuple[int, int],
    C: float | None = None,
    alpha: float | None = None,
) -> Pipeline:
    """
    Build a TF-IDF + linear classifier pipeline for event detection.

    The pipeline has two named steps:

    - ``"features"`` — a :class:`ColumnTransformer` that applies TF-IDF to
      the ``"text"`` column and drops everything else.
    - ``"classifier"`` — one of four linear classifiers controlled by
      ``model_type``.

    ``lowercase=False`` in the vectoriser because the input (``text_ml``) is
    already lowercased by the preprocessing step; normalising twice wastes
    time and is misleading.

    Parameters
    ----------
    model_type : str
        One of ``'logreg'``, ``'svm'``, ``'sgd_logloss'``, ``'cnb'``.
    seed : int
        Random state for reproducibility (ignored by ComplementNB).
    class_weight : str | dict | None
        Passed to classifiers that support it (``'balanced'`` recommended
        for imbalanced event-detection data).
    max_features : int
        Maximum vocabulary size for TF-IDF.
    min_df : int | float
        Minimum document frequency for TF-IDF.
    max_df : int | float
        Maximum document frequency for TF-IDF.
    ngram_range : tuple[int, int]
        N-gram range for TF-IDF (e.g. ``(1, 2)``).
    C : float | None
        Regularisation parameter for ``logreg`` and ``svm``.
    alpha : float | None
        Regularisation / smoothing parameter for ``sgd_logloss`` and ``cnb``.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline ready for ``.fit()``.

    Raises
    ------
    ValueError
        If ``model_type`` is not recognised or a required hyperparameter is
        missing for the chosen classifier.
    """
    if model_type not in _MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {sorted(_MODEL_TYPES)}"
        )

    # ------------------------------------------------------------------
    # TF-IDF vectoriser
    # lowercase=False: text_ml is already lowercased at preprocessing time
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        lowercase=False,
        sublinear_tf=True,   # log(1 + tf) — reduces impact of very frequent terms
    )

    features = ColumnTransformer(
        transformers=[("tfidf", vectorizer, "text")],
        remainder="drop",
    )

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------
    if model_type == "logreg":
        if C is None:
            raise ValueError("'logreg' requires C")
        clf = LogisticRegression(
            C=C,
            max_iter=5000,
            class_weight=class_weight,
            random_state=seed,
            solver="lbfgs",
        )

    elif model_type == "svm":
        if C is None:
            raise ValueError("'svm' requires C")
        clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=seed,
            max_iter=5000,
        )

    elif model_type == "sgd_logloss":
        if alpha is None:
            raise ValueError("'sgd_logloss' requires alpha")
        clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            penalty="l2",
            class_weight=class_weight,
            random_state=seed,
            max_iter=2000,
            tol=1e-3,
        )

    elif model_type == "cnb":
        if alpha is None:
            raise ValueError("'cnb' requires alpha")
        # ComplementNB does not support class_weight or random_state
        clf = ComplementNB(alpha=alpha)

    return Pipeline([("features", features), ("classifier", clf)])