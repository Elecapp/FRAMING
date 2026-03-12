# src/framing_ml/baselines.py
# TF-IDF baselines: Logistic Regression, Linear SVM, Naive Bayes.
#
# text_ml is already lowercased — lowercase=False in TfidfVectorizer.
# class_weight="balanced" throughout to handle skewed VB / PJ labels.

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Candidate configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateConfig:
    ngram_range: tuple
    max_features: int
    min_df: int
    c: float = 1.0
    alpha: float = 1.0


def get_candidate_configs(model_name: str) -> list[CandidateConfig]:
    if model_name in ("logreg", "linear_svm"):
        return [
            CandidateConfig(ngram_range=(1, 1), max_features=10_000, min_df=2, c=0.5),
            CandidateConfig(ngram_range=(1, 1), max_features=10_000, min_df=2, c=1.0),
            CandidateConfig(ngram_range=(1, 1), max_features=10_000, min_df=2, c=2.0),
            CandidateConfig(ngram_range=(1, 2), max_features=10_000, min_df=2, c=1.0),
            CandidateConfig(ngram_range=(1, 2), max_features=10_000, min_df=2, c=2.0),
            CandidateConfig(ngram_range=(1, 2), max_features=20_000, min_df=2, c=2.0),
        ]
    if model_name == "naive_bayes":
        return [
            CandidateConfig(ngram_range=(1, 1), max_features=10_000, min_df=2, alpha=0.5),
            CandidateConfig(ngram_range=(1, 1), max_features=10_000, min_df=2, alpha=1.0),
            CandidateConfig(ngram_range=(1, 2), max_features=10_000, min_df=2, alpha=0.5),
            CandidateConfig(ngram_range=(1, 2), max_features=10_000, min_df=2, alpha=1.0),
        ]
    raise ValueError(f"Unsupported model_name: {model_name}")


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(model_name: str, cfg: CandidateConfig, task_type: str) -> Pipeline:
    # text_ml is already lowercased
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        sublinear_tf=True,
    )

    if model_name == "logreg":
        solver = "liblinear" if task_type == "binary" else "lbfgs"
        clf = LogisticRegression(
            C=cfg.c,
            class_weight="balanced",
            max_iter=3000,
            solver=solver,
            random_state=RANDOM_STATE,
        )
    elif model_name == "linear_svm":
        clf = LinearSVC(
            C=cfg.c,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=3000,
        )
    elif model_name == "naive_bayes":
        # NB does not support class_weight; use sample_weight at fit time if needed
        clf = MultinomialNB(alpha=cfg.alpha)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return Pipeline(steps=[("tfidf", vectorizer), ("clf", clf)])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_binary(y_true, y_pred) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def evaluate_multiclass(y_true, y_pred) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "macro_f1":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metric_to_optimize(task_type: str) -> str:
    if task_type == "binary":
        return "f1"
    if task_type == "multiclass":
        return "macro_f1"
    raise ValueError(f"Unsupported task_type: {task_type}")


def evaluate_task(y_true, y_pred, task_type: str) -> dict:
    if task_type == "binary":
        return evaluate_binary(y_true, y_pred)
    if task_type == "multiclass":
        return evaluate_multiclass(y_true, y_pred)
    raise ValueError(f"Unsupported task_type: {task_type}")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str,
    task_type: str,
) -> dict:
    x_train = train_df[text_col].fillna("").astype(str)
    y_train = train_df[label_col]
    x_val   = val_df[text_col].fillna("").astype(str)
    y_val   = val_df[label_col]

    best_score    = -1.0
    best_cfg      = None
    best_metrics  = None
    best_pipeline = None
    search_results: list[dict] = []
    target_metric = metric_to_optimize(task_type)

    for cfg in get_candidate_configs(model_name):
        pipeline = build_pipeline(model_name, cfg, task_type)
        pipeline.fit(x_train, y_train)

        val_pred    = pipeline.predict(x_val)
        val_metrics = evaluate_task(y_val, val_pred, task_type)

        # also record train metrics to detect overfitting
        train_pred    = pipeline.predict(x_train)
        train_metrics = evaluate_task(y_train, train_pred, task_type)

        search_results.append({
            "model_name":    model_name,
            "label_col":     label_col,
            "task_type":     task_type,
            "params": {
                "ngram_range":  list(cfg.ngram_range),
                "max_features": cfg.max_features,
                "min_df":       cfg.min_df,
                "C":            cfg.c,
                "alpha":        cfg.alpha,
            },
            "train_metrics": train_metrics,
            "val_metrics":   val_metrics,
        })

        if val_metrics[target_metric] > best_score:
            best_score    = val_metrics[target_metric]
            best_cfg      = cfg
            best_metrics  = val_metrics
            best_pipeline = pipeline

    if best_cfg is None:
        raise RuntimeError("No valid configuration found during baseline tuning.")

    return {
        "best_pipeline":    best_pipeline,
        "best_config": {
            "ngram_range":  list(best_cfg.ngram_range),
            "max_features": best_cfg.max_features,
            "min_df":       best_cfg.min_df,
            "C":            best_cfg.c,
            "alpha":        best_cfg.alpha,
        },
        "best_val_metrics": best_metrics,
        "search_results":   search_results,
    }


# ---------------------------------------------------------------------------
# Final fit (train + val) and evaluation on test
# ---------------------------------------------------------------------------

def fit_final_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str,
    best_config: dict,
    task_type: str,
) -> Pipeline:
    cfg = CandidateConfig(
        ngram_range=tuple(best_config["ngram_range"]),
        max_features=int(best_config["max_features"]),
        min_df=int(best_config["min_df"]),
        c=float(best_config.get("C", 1.0)),
        alpha=float(best_config.get("alpha", 1.0)),
    )
    pipeline     = build_pipeline(model_name, cfg, task_type)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    pipeline.fit(
        train_val_df[text_col].fillna("").astype(str),
        train_val_df[label_col],
    )
    return pipeline


def evaluate_final_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
) -> tuple[dict, object]:
    x_test = test_df[text_col].fillna("").astype(str)
    y_test = test_df[label_col]
    y_pred = model.predict(x_test)
    return evaluate_task(y_test, y_pred, task_type), y_pred

