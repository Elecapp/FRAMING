# runners/run_text_preprocess.py
# Preprocess one raw CSV and write <stem>_preprocessed.csv to data/processed/.

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import pandas as pd

from src.preprocessing.text_preprocess import build_text_linguistic, build_text_ml

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

# Dataset to process — switch by commenting/uncommenting one line
INPUT = ROOT / "data" / "raw" / "EDS.csv"
# INPUT = ROOT / "data" / "raw" / "FAS.csv"
# INPUT = ROOT / "data" / "raw" / "LOC_v2.csv"

DATA_PROC = ROOT / "data" / "processed"

# Columns that must be present in every input file
_REQUIRED_COLS: list[str] = ["id", "title", "cleaned_text"]

# Optional label columns: kept when present, silently ignored otherwise
_OPTIONAL_LABEL_COLS: list[str] = [
    "label_event",        # EDS
    "victim_blaming_gt",  # FAS
    "perp_justified_gt",  # FAS
    "tone_label_gt",      # FAS
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_separator(path: Path) -> str:
    """
    Sniff the CSV delimiter from the first 4 KB of the file.

    Falls back to comma if the sniffer cannot determine the delimiter.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str
        Single-character delimiter.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        sample = fh.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        logger.info(f"Detected separator: {dialect.delimiter!r}")
        return dialect.delimiter
    except csv.Error:
        logger.warning("Could not detect separator — falling back to ','")
        return ","


def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with auto-detected delimiter."""
    return pd.read_csv(
        path,
        sep=_detect_separator(path),
        encoding="utf-8",
        keep_default_na=False,
        dtype=str,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Preprocess text from a raw CSV and save to data/processed/."""

    if not INPUT.exists():
        logger.critical(f"Input file not found: {INPUT}")
        sys.exit(1)

    logger.info(f"Loading: {INPUT}")
    df = _load_csv(INPUT)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    missing = set(_REQUIRED_COLS) - set(df.columns)
    if missing:
        logger.critical(f"Missing required columns: {missing}")
        sys.exit(1)

    # Select columns to keep
    keep = list(_REQUIRED_COLS)
    for col in _OPTIONAL_LABEL_COLS:
        if col in df.columns:
            keep.append(col)
            logger.info(f"Optional column kept: '{col}'")

    df = df[keep].copy()

    # Build normalised text representations
    logger.info("Building text_linguistic...")
    df["text_linguistic"] = [
        build_text_linguistic(title, body)
        for title, body in zip(df["title"], df["cleaned_text"])
    ]

    logger.info("Building text_ml...")
    df["text_ml"] = [
        build_text_ml(text)
        for text in df["text_linguistic"]
    ]

    # Sanity check
    n_empty_sym = (df["text_linguistic"] == "").sum()
    n_empty_ml  = (df["text_ml"] == "").sum()
    if n_empty_sym:
        logger.warning(f"{n_empty_sym} articles produced an empty text_symbolic")
    if n_empty_ml:
        logger.warning(f"{n_empty_ml} articles produced an empty text_ml")

    # Save
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROC / f"{INPUT.stem}_preprocessed.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"Saved: {output_path}")
    print(f"\n  {INPUT.name}: {output_path}  ({len(df)} rows)\n")


if __name__ == "__main__":
    main()