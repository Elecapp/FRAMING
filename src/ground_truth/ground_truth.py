# src/ground_truth/ground_truth.py
# Build final ground truth Human-Labels (H-Labels) from 3 annotators' XLSX files.
#
# Majority voting: an article receives a positive label when at least 2 out
# of 3 annotators agree.  For tone (3-class), full disagreement is possible
# and is handled explicitly.


from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns each annotator file must contain
_REQUIRED_COLS: list[str] = [
    "id",
    "title",
    "cleaned_text",
    "victim_blaming_manual",
    "perp_justified_manual",
    "tone_label_manual",
]

# Framing dimensions and their short codes used internally
_DIMENSIONS: dict[str, str] = {
    "vb": "victim_blaming",
    "pj": "perp_justified",
    "tone": "tone_label",
}

# Annotator identifiers (order must match file_a / file_b / file_c)
_ANNOTATORS: list[str] = ["a", "b", "c"]

# Pairs for pairwise agreement
_PAIRS: list[tuple[str, str]] = list(combinations(_ANNOTATORS, 2))


# → [('a', 'b'), ('a', 'c'), ('b', 'c')]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthConfig:
    """
    Paths and options for ground-truth construction.

    Parameters
    ----------
    file_a, file_b, file_c : Path
        XLSX annotation files for the three annotators.
    out_gt : Path
        CSV output path for the consensus (H-Label) dataset.
    out_agreement : Path | None
        Optional CSV output path for pairwise inter-annotator agreement stats.
    drop_no_majority : bool
        If True, rows where all three annotators disagree on tone are dropped
        with a warning.  If False, they are kept with NaN and logged.
    """

    file_a: Path
    file_b: Path
    file_c: Path
    out_gt: Path
    out_agreement: Path | None = None
    drop_no_majority: bool = True

    def __post_init__(self) -> None:
        """Convert to Path and verify that all input files exist."""
        for attr in ("file_a", "file_b", "file_c"):
            p = Path(getattr(self, attr))
            object.__setattr__(self, attr, p)
            if not p.exists():
                raise FileNotFoundError(f"Annotation file not found: {p}")

        self.out_gt = Path(self.out_gt)
        if self.out_agreement is not None:
            self.out_agreement = Path(self.out_agreement)


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

def _majority_3(a: Any, b: Any, c: Any) -> Any | None:
    """
    Return the majority value among three annotators.

    With three annotators, at least two must agree for any binary label.
    For multi-class labels (e.g. tone with 3 classes), all three can
    disagree, in which case ``None`` is returned.

    Parameters
    ----------
    a, b, c : Any
        Labels from the three annotators (int or str).

    Returns
    -------
    Any | None
        Majority label, or None when all three values are distinct.
    """
    if a == b or a == c:
        return a
    if b == c:
        return b
    # Full disagreement only possible for non-binary dimensions (e.g. tone)
    return None


# ---------------------------------------------------------------------------
# Inter-annotator agreement
# ---------------------------------------------------------------------------

def compute_agreement(df: pd.DataFrame):  # -> pd.DataFrame:
    """
    Compute pairwise Cohen's kappa for each framing dimension.

    Kappa is undefined when one annotator uses only a single label (zero
    variance).  In that case the score is recorded as NaN with a warning
    rather than raising an exception.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ``{dim}_{annot}`` for every
        dimension in ``_DIMENSIONS`` and every annotator in ``_ANNOTATORS``.

    Returns
    -------
    pd.DataFrame
        Rows: one per (dimension, annotator-pair) combination.
        Columns: dimension, pair, cohen_kappa, n_samples.
    """
    records: list[dict] = []

    for dim in _DIMENSIONS:
        for a1, a2 in _PAIRS:
            col1, col2 = f"{dim}_{a1}", f"{dim}_{a2}"

            # Drop rows where either annotator left a NaN
            mask = df[col1].notna() & df[col2].notna()
            y1 = df.loc[mask, col1].tolist()
            y2 = df.loc[mask, col2].tolist()
            n = len(y1)

            try:
                kappa = cohen_kappa_score(y1, y2)
            except ValueError as exc:
                # Raised when one sequence has only a single unique label
                logger.warning(
                    f"Cannot compute kappa for {dim} {a1}_vs_{a2} "
                    f"(n={n}): {exc}"
                )
                kappa = np.nan

            records.append({
                "dimension": dim,
                "pair": f"{a1}_vs_{a2}",
                "cohen_kappa": round(kappa, 3) if not np.isnan(kappa) else np.nan,
                "n_samples": n,
            })

    return pd.DataFrame(records)


def _print_agreement(agreement_df: pd.DataFrame) -> None:
    """Pretty-print agreement table to stdout."""
    print("\n" + "=" * 65)
    print("INTER-ANNOTATOR AGREEMENT  (Cohen's κ)")
    print("=" * 65)
    print(f"{'Dimension':<10} {'Pair':<14} {'κ':>8}  {'n':>6}")
    print("-" * 65)
    for _, row in agreement_df.iterrows():
        kappa_str = f"{row['cohen_kappa']:.3f}" if not pd.isna(row["cohen_kappa"]) else "  NaN"
        print(
            f"{row['dimension']:<10} {row['pair']:<14} "
            f"{kappa_str:>8}  {int(row['n_samples']):>6}"
        )
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_annotator(path: Path, annot: str) -> pd.DataFrame:
    """
    Read a single annotator XLSX file and rename columns to short codes.

    Parameters
    ----------
    path : Path
        Path to the XLSX file.
    annot : str
        Single-letter annotator identifier (``'a'``, ``'b'``, or ``'c'``).

    Returns
    -------
    pd.DataFrame
        Columns: id, title, cleaned_text, vb_{annot}, pj_{annot}, tone_{annot}.
    """
    logger.debug(f"Reading annotator '{annot}' from {path}")

    df = pd.read_excel(path)

    missing = set(_REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Annotator '{annot}' file {path} is missing columns: {missing}"
        )

    df = df[_REQUIRED_COLS].copy()
    df["id"] = df["id"].astype(str).str.strip()

    return df.rename(columns={
        "victim_blaming_manual": f"vb_{annot}",
        "perp_justified_manual": f"pj_{annot}",
        "tone_label_manual": f"tone_{annot}",
    })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_ground_truth(cfg: GroundTruthConfig) -> pd.DataFrame:
    """
    Build consensus GT-Labels from three independent annotators.

    Pipeline
    --------
    1. Read and validate each annotator's XLSX file.
    2. Inner-join on article ``id`` to retain only fully annotated articles.
    3. Compute pairwise inter-annotator agreement (Cohen's κ) and optionally
       persist to CSV.
    4. Apply majority voting per framing dimension.
    5. Handle full-disagreement rows for multi-class tone.
    6. Persist the consensus dataset and return it.

    Parameters
    ----------
    cfg : GroundTruthConfig

    Returns
    -------
    pd.DataFrame
        Consensus dataset with columns:
        id, title, cleaned_text,
        victim_blaming_gt, perp_justified_gt, tone_label_gt.
    """
    logger.info("Building ground truth from 3 annotators...")

    # ------------------------------------------------------------------
    # 1. Read annotator files
    # ------------------------------------------------------------------
    ann_a = _read_annotator(cfg.file_a, "a")
    ann_b = _read_annotator(cfg.file_b, "b")
    ann_c = _read_annotator(cfg.file_c, "c")

    n_a, n_b, n_c = len(ann_a), len(ann_b), len(ann_c)
    logger.info(f"Loaded articles — A: {n_a}, B: {n_b}, C: {n_c}")

    # ------------------------------------------------------------------
    # 2. Merge on article ID (inner join → only fully annotated articles)
    # ------------------------------------------------------------------
    ann_cols_b = ["id", "vb_b", "pj_b", "tone_b"]
    ann_cols_c = ["id", "vb_c", "pj_c", "tone_c"]

    df = (
        ann_a
        .merge(ann_b[ann_cols_b], on="id", how="inner")
        .merge(ann_c[ann_cols_c], on="id", how="inner")
    )

    n_merged = len(df)
    if n_merged < min(n_a, n_b, n_c):
        logger.warning(
            f"Inner merge reduced dataset from min({n_a},{n_b},{n_c})="
            f"{min(n_a, n_b, n_c)} to {n_merged}. "
            "Check for mismatched article IDs across annotator files."
        )
    logger.info(f"Merged dataset: {n_merged} articles")

    # ------------------------------------------------------------------
    # 3. Inter-annotator agreement
    # ------------------------------------------------------------------
    logger.info("Computing inter-annotator agreement (Cohen's κ)...")
    agreement_df = compute_agreement(df)
    _print_agreement(agreement_df)

    if cfg.out_agreement is not None:
        cfg.out_agreement.parent.mkdir(parents=True, exist_ok=True)
        agreement_df.to_csv(cfg.out_agreement, index=False)
        logger.info(f"Agreement stats saved: {cfg.out_agreement}")

    # ------------------------------------------------------------------
    # 4. Majority voting per dimension
    # ------------------------------------------------------------------
    logger.info("Applying majority voting...")

    for dim, out_col in [
        ("vb", "victim_blaming_gt"),
        ("pj", "perp_justified_gt"),
        ("tone", "tone_label_gt"),
    ]:
        df[out_col] = [
            _majority_3(a, b, c)
            for a, b, c in zip(df[f"{dim}_a"], df[f"{dim}_b"], df[f"{dim}_c"])
        ]

    # ------------------------------------------------------------------
    # 5. Handle full-disagreement rows (None values from _majority_3)
    # ------------------------------------------------------------------
    no_majority_mask = df["tone_label_gt"].isna()
    n_no_majority = no_majority_mask.sum()

    if n_no_majority > 0:
        ids = df.loc[no_majority_mask, "id"].tolist()
        logger.warning(
            f"{n_no_majority} article(s) have no tone majority "
            f"(all 3 annotators disagree): {ids}"
        )
        if cfg.drop_no_majority:
            df = df[~no_majority_mask].copy()
            logger.warning(
                f"Dropped {n_no_majority} row(s) with no tone majority "
                "(set drop_no_majority=False to keep them as NaN)."
            )
        # else: rows are kept with NaN — callers must handle downstream

    # ------------------------------------------------------------------
    # 6. Select output columns and log statistics
    # ------------------------------------------------------------------
    out_cols = [
        "id", "title", "cleaned_text",
        "victim_blaming_gt",
        "perp_justified_gt",
        "tone_label_gt",
    ]
    out = df[out_cols].copy()

    logger.info(f"Ground truth built: {len(out)} articles")
    logger.info(f"  Victim blaming  — positive: {out['victim_blaming_gt'].sum()}")
    logger.info(f"  Perp justified  — positive: {out['perp_justified_gt'].sum()}")
    logger.info(
        f"  Tone distribution:\n"
        + out["tone_label_gt"].value_counts(dropna=False).to_string()
    )

    # ------------------------------------------------------------------
    # 7. Persist
    # ------------------------------------------------------------------
    cfg.out_gt.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.out_gt, index=False)
    logger.info(f"Saved ground truth: {cfg.out_gt}")

    return out
