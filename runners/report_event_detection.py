# runners/report_event_detection.py
# EDS statistics and event detection model performance.
# Reads from EDS_preprocessed.csv and the reports produced by
# run_event_detection.py.
#
# Output: reports/paper_numbers/event_detection.json

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

EDS_PATH          = ROOT / "data"    / "processed" / "EDS_preprocessed.csv"
TEST_PREDS_PATH   = ROOT / "data"    / "processed" / "EDS_test_predictions.csv"
VAL_REPORT_PATH   = ROOT / "reports" / "event_detection" / "validation_report.txt"
TEST_REPORT_PATH  = ROOT / "reports" / "event_detection" / "test_report.txt"
CONFIG_PATH       = ROOT / "reports" / "event_detection" / "config.json"
OUT_DIR           = ROOT / "reports" / "paper_numbers"

LABEL_COL = "label_event"


def _class_dist(series: pd.Series) -> dict:
    vc      = series.value_counts()
    vc_norm = series.value_counts(normalize=True)
    return {
        "counts":      {str(k): int(v)   for k, v in vc.items()},
        "proportions": {str(k): round(float(v), 4) for k, v in vc_norm.items()},
    }


def _read_report_txt(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {}

    # ------------------------------------------------------------------
    # 1. EDS size and class distribution
    # ------------------------------------------------------------------
    if not EDS_PATH.exists():
        logger.critical(f"EDS preprocessed file not found: {EDS_PATH}")
        sys.exit(1)

    eds = pd.read_csv(EDS_PATH, keep_default_na=False)
    report["eds"] = {
        "total_articles": int(len(eds)),
    }

    if LABEL_COL in eds.columns:
        dist = _class_dist(eds[LABEL_COL])
        report["eds"]["label_distribution"] = dist
        report["eds"]["n_positive"] = dist["counts"].get("1", dist["counts"].get(1, None))
        report["eds"]["n_negative"] = dist["counts"].get("0", dist["counts"].get(0, None))
        logger.info(
            f"EDS: {len(eds)} articles — "
            f"positive={report['eds']['n_positive']}  "
            f"negative={report['eds']['n_negative']}"
        )
    else:
        logger.warning(f"Column '{LABEL_COL}' not found in EDS — no class distribution")

    # ------------------------------------------------------------------
    # 2. Split sizes (inferred from test predictions + full EDS)
    #    The runner uses 60/20/20 stratified split.
    # ------------------------------------------------------------------
    n_total = len(eds)
    report["eds"]["splits"] = {
        "strategy":   "stratified 60/20/20",
        "n_train":    round(n_total * 0.60),
        "n_val":      round(n_total * 0.20),
        "n_test":     round(n_total * 0.20),
    }

    # If test predictions are available, use actual counts
    if TEST_PREDS_PATH.exists():
        test_preds = pd.read_csv(TEST_PREDS_PATH, keep_default_na=False)
        report["eds"]["splits"]["n_test_actual"] = int(len(test_preds))
        logger.info(f"Test predictions: {len(test_preds)} articles")

    # ------------------------------------------------------------------
    # 3. Best model config
    # ------------------------------------------------------------------
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            config = json.load(fh)
        report["best_model_config"] = config
        logger.info(f"Best model config loaded: {config}")
    else:
        logger.warning(f"Config not found: {CONFIG_PATH}")
        report["best_model_config"] = None

    # ------------------------------------------------------------------
    # 4. Validation and test metrics
    #    Stored as raw text reports — include verbatim for reference,
    #    and also parse key metrics for programmatic use.
    # ------------------------------------------------------------------
    report["validation_report"] = _read_report_txt(VAL_REPORT_PATH)
    report["test_report"]       = _read_report_txt(TEST_REPORT_PATH)

    # Parse precision / recall / f1 / threshold from test predictions
    if TEST_PREDS_PATH.exists():
        test_preds = pd.read_csv(TEST_PREDS_PATH, keep_default_na=False)

        if {"label_event", "pred_event"}.issubset(test_preds.columns):
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                confusion_matrix,
            )
            y_true = test_preds["label_event"].astype(int)
            y_pred = test_preds["pred_event"].astype(int)

            cm = confusion_matrix(y_true, y_pred)
            report["test_metrics"] = {
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
                "confusion_matrix": {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                },
            }
            logger.info(f"Test metrics: {report['test_metrics']}")
        else:
            logger.warning("Columns label_event / pred_event not found in test predictions")
            report["test_metrics"] = None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = OUT_DIR / "event_detection.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved : {out_path}")

    print(f"\n -  Event detection report saved : {out_path}\n")


if __name__ == "__main__":
    main()

