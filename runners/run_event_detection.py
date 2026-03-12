# runners/run_event_detection.py
# Train the event-detection classifier, evaluate on held-out test set,
# and apply to the full LOC corpus to produce event-detection scores.
#
# Pipeline
# --------
#   1. Load EDS (annotated subset) and LOC (full local-news corpus).
#   2. Stratified 60 / 20 / 20 split on EDS.
#   3. Grid search over model types and hyperparameters; select by val F1.
#   4. Retrain the best config on train + val.
#   5. Evaluate on held-out test set : data/processed/EDS_test_predictions.csv
#   6. Apply to LOC : data/processed/LOC_v2_event_detection.csv

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.event_detection.evaluation import (
    get_scores,
    predict_with_threshold,
    tune_threshold_f1,
)
from src.event_detection.model import build_event_classifier
from src.utils.io import ensure_dir, write_json
from src.utils.metrics import binary_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# Input datasets (preprocessed)
TRAIN_INPUT = ROOT / "data" / "processed" / "EDS_preprocessed.csv"
LOC_INPUT = ROOT / "data" / "processed" / "LOC_v2_preprocessed.csv"

# Future: when LOC manual annotations are available, move LOC_v2_annotated.csv
# to data/raw/ and run preprocessing to produce LOC_v2_annotated_preprocessed.csv
# LOC_ANNOTATED_INPUT = ROOT / "data" / "processed" / "LOC_v2_annotated_preprocessed.csv"

# Column names
TEXT_COL = "text_ml"
LABEL_COL = "label_event"

# Columns forwarded to downstream framing modules
KEEP_COLS = ["id", "title", "cleaned_text", "text_linguistic"]

# Experiment
SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# TF-IDF
MAX_FEATURES = 50_000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.9

# Hyperparameter grids
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise ValueError if any required column is missing from df."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _append_report_section(path: Path, header: str, content: str) -> None:
    """Append a labelled section to a plain-text report file."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 80}\n{header}\n{'-' * 80}\n{content.strip()}\n")


def _candidate_configs() -> list[tuple[str, str, dict]]:
    """Return all (model_type, label, params) configurations to evaluate."""
    configs = []
    for C in C_GRID:
        configs.append(("logreg", f"C={C}", {"C": C}))
        configs.append(("svm", f"C={C}", {"C": C}))
    for alpha in ALPHA_GRID:
        configs.append(("sgd_logloss", f"alpha={alpha}", {"alpha": alpha}))
        configs.append(("cnb", f"alpha={alpha}", {"alpha": alpha}))
    return configs


def _build_and_fit(model_type: str, params: dict, X_train, y_train) -> object:
    """Instantiate and fit a classifier pipeline."""
    return build_event_classifier(
        model_type=model_type,
        seed=SEED,
        class_weight="balanced",
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        **params,
    ).fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    for path, name in [(TRAIN_INPUT, "EDS"), (LOC_INPUT, "LOC")]:
        if not path.exists():
            logger.critical(f"{name} input not found: {path}")
            sys.exit(1)

    logger.info(f"Loading EDS: {TRAIN_INPUT.name}")
    eds_df = pd.read_csv(TRAIN_INPUT, keep_default_na=False)

    logger.info(f"Loading LOC: {LOC_INPUT.name}")
    loc_df = pd.read_csv(LOC_INPUT, keep_default_na=False)

    _validate_columns(eds_df, [TEXT_COL, LABEL_COL], "EDS")
    _validate_columns(loc_df, [TEXT_COL], "LOC")

    # ------------------------------------------------------------------
    # 2. Prepare arrays
    # ------------------------------------------------------------------
    X_all = pd.DataFrame({"text": eds_df[TEXT_COL].values})
    y_all = eds_df[LABEL_COL].astype(float).astype(int).values

    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    logger.info(f"EDS: {len(y_all)} articles  |  pos={n_pos}  neg={n_neg}")

    # Stratified 60 / 20 / 20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all,
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_frac,
        random_state=SEED,
        stratify=y_temp,
    )
    logger.info(
        f"Split — train: {len(y_train)}  val: {len(y_val)}  test: {len(y_test)}"
    )

    # ------------------------------------------------------------------
    # 3. Output directories
    # ------------------------------------------------------------------
    model_dir = ROOT / "models" / "event_detection"
    report_dir = ROOT / "reports" / "event_detection"
    output_dir = ROOT / "data" / "processed"

    for d in (model_dir, report_dir, output_dir):
        ensure_dir(d)

    val_report_path = report_dir / "validation_report.txt"
    test_report_path = report_dir / "test_report.txt"

    val_report_path.write_text(
        "EVENT DETECTION — VALIDATION REPORT\n", encoding="utf-8"
    )
    test_report_path.write_text(
        "EVENT DETECTION — TEST REPORT\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 4. Model selection (validation set)
    # ------------------------------------------------------------------
    logger.info("Starting model selection on validation set...")
    best: dict | None = None

    for model_type, param_str, params in _candidate_configs():
        logger.info(f"  {model_type:14s}  {param_str}")

        model = _build_and_fit(model_type, params, X_train, y_train)

        scores_val, score_type = get_scores(model, X_val)

        if scores_val is not None:
            threshold, _ = tune_threshold_f1(y_val, scores_val)
        else:
            threshold = None

        if threshold is not None:
            y_pred_val = predict_with_threshold(scores_val, threshold)
        else:
            y_pred_val = model.predict(X_val).astype(int)
            scores_val = None

        metrics_val = binary_metrics(y_val, y_pred_val, y_score=scores_val)
        f1_val = float(metrics_val["f1"])

        header = (
            f"{model_type} | {param_str} | "
            f"score={score_type} | threshold={threshold}"
        )
        _append_report_section(
            val_report_path, header,
            classification_report(y_val, y_pred_val, digits=4, zero_division=0),
        )
        _append_report_section(
            val_report_path, f"{header} | CONFUSION MATRIX",
            f"[[TN  FP]\n [FN  TP]]\n"
            f"{confusion_matrix(y_val, y_pred_val, labels=[0, 1])}",
        )

        if best is None or f1_val > best["val_f1"]:
            best = {
                "model_type": model_type,
                "param_str": param_str,
                "params": params,
                "score_type": score_type,
                "threshold": threshold,
                "val_f1": f1_val,
                "val_metrics": metrics_val,
            }
            logger.info(f"    :  new best  F1={f1_val:.4f}")

    logger.info(
        f"Best config: {best['model_type']} {best['param_str']}  "
        f"val_F1={best['val_f1']:.4f}"
    )

    # ------------------------------------------------------------------
    # 5. Retrain on train + val, evaluate on test
    # ------------------------------------------------------------------
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = np.concatenate([y_train, y_val])

    logger.info("Retraining best config on train + val...")
    final_model = _build_and_fit(best["model_type"], best["params"], X_trainval, y_trainval)

    logger.info("Evaluating on held-out test set...")
    scores_test, _ = get_scores(final_model, X_test)
    threshold = best["threshold"]

    if scores_test is not None and threshold is not None:
        y_pred_test = predict_with_threshold(scores_test, threshold)
        metrics_test = binary_metrics(y_test, y_pred_test, y_score=scores_test)
    else:
        y_pred_test = final_model.predict(X_test).astype(int)
        metrics_test = binary_metrics(y_test, y_pred_test, y_score=None)

    test_header = (
        f"FINAL: {best['model_type']} | {best['param_str']} | "
        f"threshold={threshold}"
    )
    _append_report_section(
        test_report_path, test_header,
        classification_report(y_test, y_pred_test, digits=4, zero_division=0),
    )
    _append_report_section(
        test_report_path, f"{test_header} | CONFUSION MATRIX",
        f"[[TN  FP]\n [FN  TP]]\n"
        f"{confusion_matrix(y_test, y_pred_test, labels=[0, 1])}",
    )

    logger.info(
        f"Test — P={metrics_test['precision']:.4f}  "
        f"R={metrics_test['recall']:.4f}  "
        f"F1={metrics_test['f1']:.4f}  "
        f"AUC={metrics_test.get('roc_auc', float('nan')):.4f}"
    )

    # Save EDS test predictions
    eds_test_idx = X_test.index
    eds_test_out = eds_df.loc[eds_test_idx, [c for c in KEEP_COLS if c in eds_df.columns]].copy()
    eds_test_out["label_event"] = y_test
    eds_test_out["pred_event"] = y_pred_test
    if scores_test is not None:
        eds_test_out["score_event"] = scores_test

    eds_test_out.to_csv(output_dir / "EDS_test_predictions.csv", index=False)
    logger.info(f"EDS test predictions saved :  {output_dir / 'EDS_test_predictions.csv'}")

    # ------------------------------------------------------------------
    # 6. Persist model and config
    # ------------------------------------------------------------------
    joblib.dump(final_model, model_dir / "model.joblib")
    logger.info(f"Model saved :  {model_dir / 'model.joblib'}")

    write_json(report_dir / "config.json", {
        "inputs": {
            "train": str(TRAIN_INPUT),
            "loc": str(LOC_INPUT),
        },
        "seed": SEED,
        "split": {"train": 0.60, "val": 0.20, "test": 0.20},
        "tfidf": {
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
            "max_df": MAX_DF,
        },
        "grids": {"C": C_GRID, "alpha": ALPHA_GRID},
        "best_model": {
            "type": best["model_type"],
            "params": best["param_str"],
            "threshold": threshold,
            "score_type": best["score_type"],
        },
        "val_metrics": best["val_metrics"],
        "test_metrics": metrics_test,
    })

    # ------------------------------------------------------------------
    # 7. Apply to full LOC corpus
    # ------------------------------------------------------------------
    logger.info("Applying final model to LOC corpus...")
    X_loc = pd.DataFrame({"text": loc_df[TEXT_COL].values})

    scores_loc, _ = get_scores(final_model, X_loc)

    if scores_loc is not None and threshold is not None:
        loc_df["score_event"] = scores_loc
        loc_df["pred_event"] = predict_with_threshold(scores_loc, threshold)
    else:
        loc_df["pred_event"] = final_model.predict(X_loc).astype(int)

    forward_cols = [c for c in KEEP_COLS if c in loc_df.columns]
    score_col = ["score_event"] if "score_event" in loc_df.columns else []
    output_cols = forward_cols + ["pred_event"] + score_col

    loc_out = loc_df[output_cols].copy()
    loc_out.to_csv(output_dir / "LOC_v2_event_detection.csv", index=False)

    n_pos_loc = int((loc_out["pred_event"] == 1).sum())
    logger.info(
        f"LOC — total: {len(loc_out)}  "
        f"positives: {n_pos_loc}  "
        f"({100 * n_pos_loc / len(loc_out):.1f}%)"
    )

    print(
        f"\n -  Event detection complete\n"
        f"   Best : {best['model_type']} {best['param_str']}\n"
        f"   Test F1 : {metrics_test['f1']:.4f}\n"
        f"   LOC positives : {n_pos_loc} / {len(loc_out)}\n"
    )


if __name__ == "__main__":
    main()
