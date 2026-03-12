# runners/run_build_ground_truth.py
# Entry point for building the consensus ground truth (H-Labels) from the
# three annotator XLSX files.

import logging
import sys
from pathlib import Path

from src.ground_truth.ground_truth import GroundTruthConfig, build_ground_truth

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data"


def main() -> None:
    """Build consensus Human-Labels (H-Labels) from three independent annotator files."""

    cfg = GroundTruthConfig(
        file_a=DATA_RAW / "FAS_annotation_B.xlsx",  # annotator b
        file_b=DATA_RAW / "FAS_annotation_A.xlsx",  # annotator a
        file_c=DATA_RAW / "FAS_annotation_C.xlsx",  # annotator c
        out_gt=DATA_RAW / "FAS.csv",
        out_agreement=DATA_PROC / "inter_annotator_agreement.csv",
        # drop_no_majority=True: rows where all 3 annotators disagree on tone
        # are dropped and logged.  Set False to keep them as NaN.
        drop_no_majority=True,
    )

    try:
        df = build_ground_truth(cfg)
    except FileNotFoundError as exc:
        logging.critical(str(exc))
        sys.exit(1)
    except Exception as exc:
        logging.critical(f"Unexpected error: {exc}", exc_info=True)
        sys.exit(1)

    print(f"\n-  Ground truth built successfully: {len(df)} articles")
    print(f"-   Labels:  {cfg.out_gt}")
    print(f"-   Agreement: {cfg.out_agreement}\n")


if __name__ == "__main__":
    main()
