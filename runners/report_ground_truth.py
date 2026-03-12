# runners/report_ground_truth.py
# FAS statistics: size, class distribution, inter-annotator agreement.
# Reads from FAS_preprocessed.csv and the agreement report produced by
# run_build_ground_truth.py.
#
# Output: reports/paper_numbers/ground_truth.json

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

FAS_PATH       = ROOT / "data"    / "processed" / "FAS_preprocessed.csv"
GT_REPORT_PATH = ROOT / "data" / "inter_annotator_agreement.csv"
OUT_DIR        = ROOT / "reports" / "paper_numbers"

LABEL_COLS = {
    "victim_blaming":            "victim_blaming_gt",
    "perpetrator_justification": "perp_justified_gt",
    "tone":                      "tone_label_gt",
}


def _class_dist(series: pd.Series) -> dict:
    vc      = series.value_counts()
    vc_norm = series.value_counts(normalize=True)
    return {
        "counts":      {str(k): int(v)   for k, v in vc.items()},
        "proportions": {str(k): round(float(v), 4) for k, v in vc_norm.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {}

    # ------------------------------------------------------------------
    # 1. FAS size and class distribution
    # ------------------------------------------------------------------
    if not FAS_PATH.exists():
        logger.critical(f"FAS not found: {FAS_PATH}")
        sys.exit(1)

    fas = pd.read_csv(FAS_PATH, keep_default_na=False)
    report["fas"] = {
        "total_articles": int(len(fas)),
        "label_distribution": {
            task: _class_dist(fas[col])
            for task, col in LABEL_COLS.items()
            if col in fas.columns
        },
    }
    logger.info(f"FAS: {len(fas)} articles")

    # ------------------------------------------------------------------
    # 2. Inter-annotator agreement
    # ------------------------------------------------------------------
    if not GT_REPORT_PATH.exists():
        logger.warning(
            f"Agreement report not found: {GT_REPORT_PATH} — "
            "run run_build_ground_truth.py first."
        )
        report["inter_annotator_agreement"] = None
    else:
        agr_df = pd.read_csv(GT_REPORT_PATH, keep_default_na=False)
        # CSV has columns: dimension, pair, cohen_kappa
        agreement: dict = {}
        for _, row in agr_df.iterrows():
            task  = str(row.get("dimension", ""))
            pair  = str(row.get("pair", ""))
            kappa = row.get("cohen_kappa")
            if task not in agreement:
                agreement[task] = {}
            agreement[task][pair] = {
                "kappa": round(float(kappa), 4) if kappa not in ("", None) else None,
            }

        report["inter_annotator_agreement"] = agreement
        logger.info("Inter-annotator agreement loaded")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = OUT_DIR / "ground_truth.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved : {out_path}")

    print(f"\n-  Ground truth report saved : {out_path}\n")


if __name__ == "__main__":
    main()