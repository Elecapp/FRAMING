# runners/prepare_framing_ml_splits.py
# Prepare train / val / test splits from FAS for ML framing models.
#
# Splits are stratified on tone_label_gt (multiclass, most constrained label).
# Output: data/processed/ml_splits/{fas_train, fas_val, fas_test}.csv
#         reports/framing_ml/split_report.json

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# Ground-truth annotated subset
FAS_PATH = ROOT / "data" / "processed" / "FAS_preprocessed.csv"

OUT_DIR = ROOT / "data" / "processed" / "ml_splits"
REPORT_DIR = ROOT / "reports" / "framing_ml"

ID_COL = "id"
LABEL_TONE = "tone_label_gt"  # stratification key (most constrained)

RANDOM_STATE = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load FAS
    # ------------------------------------------------------------------
    if not FAS_PATH.exists():
        logger.critical(f"FAS not found: {FAS_PATH}")
        sys.exit(1)

    fas = pd.read_csv(FAS_PATH, keep_default_na=False)
    fas = fas.drop_duplicates(subset=[ID_COL]).reset_index(drop=True)
    logger.info(f"FAS loaded: {len(fas)} articles")

    # ------------------------------------------------------------------
    # 2. Stratified split on FAS only
    # ------------------------------------------------------------------
    train_df, temp_df = train_test_split(
        fas,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=fas[LABEL_TONE],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[LABEL_TONE],
    )

    train_df.to_csv(OUT_DIR / "fas_train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUT_DIR / "fas_val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUT_DIR / "fas_test.csv", index=False, encoding="utf-8")

    logger.info(f"Split : train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Label distribution report
    # ------------------------------------------------------------------
    def _dist(df: pd.DataFrame, col: str) -> dict:
        vc = df[col].value_counts(normalize=True)
        return {k: round(float(v), 3) for k, v in vc.items()}

    report = {
        "n_fas": len(fas),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "label_distribution": {
            "train": {
                "victim_blaming_gt": _dist(train_df, "victim_blaming_gt"),
                "perp_justified_gt": _dist(train_df, "perp_justified_gt"),
                "tone_label_gt": _dist(train_df, "tone_label_gt"),
            },
            "val": {
                "victim_blaming_gt": _dist(val_df, "victim_blaming_gt"),
                "perp_justified_gt": _dist(val_df, "perp_justified_gt"),
                "tone_label_gt": _dist(val_df, "tone_label_gt"),
            },
            "test": {
                "victim_blaming_gt": _dist(test_df, "victim_blaming_gt"),
                "perp_justified_gt": _dist(test_df, "perp_justified_gt"),
                "tone_label_gt": _dist(test_df, "tone_label_gt"),
            },
        },
    }

    with open(REPORT_DIR / "split_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info(f"Split report saved : {REPORT_DIR / 'split_report.json'}")

    print(
        f"\n-  Splits ready\n"
        f"   train={len(train_df)}  val={len(val_df)}  test={len(test_df)}\n"
        f"   Output: {OUT_DIR}\n"
    )


if __name__ == "__main__":
    main()
