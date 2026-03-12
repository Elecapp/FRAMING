# runners/report_datasets.py
# Descriptive statistics for EDS, FAS, and LOC_v2.
# Run after run_text_preprocess.py.
#
# Output: reports/paper_numbers/datasets.json

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

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "paper_numbers"

EDS_PATH = ROOT / "data" / "processed" / "EDS_preprocessed.csv"
FAS_PATH = ROOT / "data" / "processed" / "FAS_preprocessed.csv"
LOC_PATH = ROOT / "data" / "processed" / "LOC_v2_preprocessed.csv"

FAS_LABEL_COLS = {
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

    # EDS
    if not EDS_PATH.exists():
        logger.critical(f"EDS not found: {EDS_PATH}")
        sys.exit(1)
    eds = pd.read_csv(EDS_PATH, keep_default_na=False)
    label = eds["label_event"].astype(float).astype(int) if "label_event" in eds.columns else None
    report["eds"] = {
        "total":      int(len(eds)),
        "n_positive": int((label == 1).sum()) if label is not None else None,
        "n_negative": int((label == 0).sum()) if label is not None else None,
    }
    logger.info(f"EDS: {report['eds']}")

    # FAS
    if not FAS_PATH.exists():
        logger.critical(f"FAS not found: {FAS_PATH}")
        sys.exit(1)
    fas = pd.read_csv(FAS_PATH, keep_default_na=False)
    report["fas"] = {
        "total": int(len(fas)),
        "label_distribution": {
            task: _class_dist(fas[col])
            for task, col in FAS_LABEL_COLS.items()
            if col in fas.columns
        },
    }
    logger.info(f"FAS: {len(fas)} articles")

    # LOC_v2
    if not LOC_PATH.exists():
        logger.critical(f"LOC_v2 not found: {LOC_PATH}")
        sys.exit(1)
    loc = pd.read_csv(LOC_PATH, keep_default_na=False)
    report["loc_v2"] = {
        "total": int(len(loc)),
    }
    logger.info(f"LOC_v2: {len(loc)} articles")

    out_path = OUT_DIR / "datasets.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()