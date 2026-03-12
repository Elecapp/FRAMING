# src/framing_ml/xgb_models.py
# XGBoost classifier on TF-IDF features.
#
# text_ml is already lowercased — lowercase=False in TfidfVectorizer.

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Candidate configs
# ---------------------------------------------------------------------------

def get_xgb_candidate_configs() -> list[dict]:
    return [
        {"ngram_range": (1, 1), "max_features": 5_000, "min_df": 2,
         "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"ngram_range": (1, 2), "max_features": 5_000, "min_df": 2,
         "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"ngram_range": (1, 1), "max_features": 10_000, "min_df": 2,
         "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
        {"ngram_range": (1, 2), "max_features": 10_000, "min_df": 2,
         "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    ]


def build_label_mapping(series: pd.Series) -> tuple[dict, dict]:
    labels  = sorted(series.dropna().unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_xgb_pipeline(cfg: dict, task_type: str, num_classes: int | None = None) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=cfg["ngram_range"],
        max_features=cfg["max_features"],
        min_df=cfg["min_df"],
        sublinear_tf=True,
    )

    common = dict(
        random_state=RANDOM_STATE,
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_jobs=1,
        tree_method="hist",
    )

    if task_type == "binary":
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    elif task_type == "multiclass":
        if num_classes is None:
            raise ValueError("num_classes required for multiclass XGBoost")
        clf = XGBClassifier(
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="mlogloss",
            **common,
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return Pipeline(steps=[("tfidf", vectorizer), ("clf", clf)])


# ---------------------------------------------------------------------------
# Evaluation helpers (consistent with baselines.py)
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
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metric_to_optimize(task_type: str) -> str:
    return "f1" if task_type == "binary" else "macro_f1"


def evaluate_task(y_true, y_pred, task_type: str) -> dict:
    if task_type == "binary":
        return evaluate_binary(y_true, y_pred)
    if task_type == "multiclass":
        return evaluate_multiclass(y_true, y_pred)
    raise ValueError(f"Unsupported task_type: {task_type}")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
) -> dict:
    x_train = train_df[text_col].fillna("").astype(str)
    x_val   = val_df[text_col].fillna("").astype(str)

    if task_type == "multiclass":
        combined       = pd.concat([train_df[label_col], val_df[label_col]], ignore_index=True)
        label2id, id2label = build_label_mapping(combined)
        y_train    = train_df[label_col].map(label2id)
        y_val      = val_df[label_col].map(label2id)
        num_classes = len(label2id)
    else:
        label2id = id2label = None
        y_train    = train_df[label_col]
        y_val      = val_df[label_col]
        num_classes = None

    best_score    = -1.0
    best_cfg      = None
    best_metrics  = None
    best_pipeline = None
    search_results: list[dict] = []
    target_metric = metric_to_optimize(task_type)

    for cfg in get_xgb_candidate_configs():
        pipeline = build_xgb_pipeline(cfg, task_type, num_classes=num_classes)
        pipeline.fit(x_train, y_train)

        raw_val   = pipeline.predict(x_val)
        raw_train = pipeline.predict(x_train)

        if task_type == "multiclass":
            val_pred   = [id2label[int(x)] for x in raw_val]
            train_pred = [id2label[int(x)] for x in raw_train]
            val_metrics   = evaluate_task(val_df[label_col],   val_pred,   task_type)
            train_metrics = evaluate_task(train_df[label_col], train_pred, task_type)
        else:
            val_pred      = raw_val
            train_pred    = raw_train
            val_metrics   = evaluate_task(y_val,   val_pred,   task_type)
            train_metrics = evaluate_task(y_train, train_pred, task_type)

        search_results.append({
            "model_name":  "xgboost",
            "label_col":   label_col,
            "task_type":   task_type,
            "params": {
                "ngram_range":  list(cfg["ngram_range"]),
                "max_features": cfg["max_features"],
                "min_df":       cfg["min_df"],
                "n_estimators": cfg["n_estimators"],
                "max_depth":    cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
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
        raise RuntimeError("No valid configuration found during XGBoost tuning.")

    return {
        "best_pipeline":    best_pipeline,
        "best_config": {
            "ngram_range":   list(best_cfg["ngram_range"]),
            "max_features":  best_cfg["max_features"],
            "min_df":        best_cfg["min_df"],
            "n_estimators":  best_cfg["n_estimators"],
            "max_depth":     best_cfg["max_depth"],
            "learning_rate": best_cfg["learning_rate"],
        },
        "best_val_metrics": best_metrics,
        "search_results":   search_results,
        "label2id":         label2id,
        "id2label":         id2label,
    }


# ---------------------------------------------------------------------------
# Final fit and evaluation on test
# ---------------------------------------------------------------------------

def fit_final_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
    best_config: dict,
) -> tuple[Pipeline, dict | None, dict | None]:
    cfg = {
        "ngram_range":   tuple(best_config["ngram_range"]),
        "max_features":  int(best_config["max_features"]),
        "min_df":        int(best_config["min_df"]),
        "n_estimators":  int(best_config["n_estimators"]),
        "max_depth":     int(best_config["max_depth"]),
        "learning_rate": float(best_config["learning_rate"]),
    }

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    x_train_val  = train_val_df[text_col].fillna("").astype(str)

    if task_type == "multiclass":
        label2id, id2label = build_label_mapping(train_val_df[label_col])
        y_train_val = train_val_df[label_col].map(label2id)
        num_classes = len(label2id)
    else:
        label2id = id2label = None
        y_train_val = train_val_df[label_col]
        num_classes = None

    pipeline = build_xgb_pipeline(cfg, task_type, num_classes=num_classes)
    pipeline.fit(x_train_val, y_train_val)
    return pipeline, label2id, id2label


def evaluate_final_xgb(
    model: Pipeline,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    task_type: str,
    id2label: dict | None = None,
) -> tuple[dict, object]:
    x_test = test_df[text_col].fillna("").astype(str)
    y_test = test_df[label_col]
    y_pred = model.predict(x_test)

    if task_type == "multiclass":
        if id2label is None:
            raise ValueError("id2label required for multiclass XGBoost evaluation.")
        y_pred = [id2label[int(x)] for x in y_pred]

    return evaluate_task(y_test, y_pred, task_type), y_pred