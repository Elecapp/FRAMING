# runners/analyze_framing_dimensions.py
# Descriptive analysis of framing dimensions (VB, PJ, tone) on unseen corpora.
#
# Intended for: LOC_v2_event_detection_framing.csv or similar unannotated data.
# Do NOT use for evaluation — performance reports are in reports/rule_based_tuning/.
#
# Outputs (reports/framing_dimensions/):
#   dimensions.json         — full results
#   dimensions_stats.csv    — flat stats table
#   dimensions_agreement.csv — inter-method agreement (rate + κ)
#   correlations.csv        — pairwise correlations between dimensions
#   summary.txt             — human-readable report

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

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

INPUT      = ROOT / "data" / "processed" / "LOC_v2_event_detection_framing.csv"
OUTPUT_DIR = ROOT / "reports" / "framing_dimensions"

# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

_REQUIRED_COLS: list[str] = [
    "victim_blaming_rulebased",
    "victim_blaming_regex",
    "victim_blaming_spacy",
    "victim_blaming_stanza",
    "victim_blaming_ratio_spacy",
    "perp_justified_rulebased",
    "perp_justified_regex",
    "perp_justified_spacy",
    "perp_justified_stanza",
    "just_ratio_spacy",
    "tone_rulebased",
    "tone_regex",
    "tone_spacy",
    "tone_stanza",
    "tone_emo_ratio_spacy",
    "tone_sens_ratio_spacy",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_kappa(a: pd.Series, b: pd.Series) -> float | None:
    try:
        return float(cohen_kappa_score(a, b))
    except ValueError:
        return None


def _agreement_pair(a: pd.Series, b: pd.Series) -> dict:
    return {
        "rate":  float((a == b).mean()),
        "kappa": _safe_kappa(a, b),
    }


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_dimension_stats(df: pd.DataFrame) -> dict:
    """Descriptive statistics for VB, PJ, and tone."""

    def _binary(label_col: str, ratio_col: str) -> dict:
        labels = df[label_col]
        ratios = df[ratio_col]
        return {
            "total_positive": int(labels.sum()),
            "total_negative": int((~labels.astype(bool)).sum()),
            "rate":           float(labels.mean()),
            "ratio_mean":     float(ratios.mean()),
            "ratio_std":      float(ratios.std()),
            "ratio_median":   float(ratios.median()),
            "ratio_min":      float(ratios.min()),
            "ratio_max":      float(ratios.max()),
        }

    def _class(col: str) -> dict:
        vc      = df[col].value_counts()
        vc_norm = df[col].value_counts(normalize=True)
        return {
            "counts":       {k: int(v)   for k, v in vc.items()},
            "distribution": {k: float(v) for k, v in vc_norm.items()},
        }

    return {
        "victim_blaming": _binary("victim_blaming_rulebased", "victim_blaming_ratio_spacy"),
        "justification":  _binary("perp_justified_rulebased", "just_ratio_spacy"),
        "tone": {
            **_class("tone_rulebased"),
            "emo_ratio_mean":  float(df["tone_emo_ratio_spacy"].mean()),
            "emo_ratio_std":   float(df["tone_emo_ratio_spacy"].std()),
            "sens_ratio_mean": float(df["tone_sens_ratio_spacy"].mean()),
            "sens_ratio_std":  float(df["tone_sens_ratio_spacy"].std()),
        },
    }


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Pairwise correlations between framing dimensions.
    Binary×binary → phi coefficient via pandas .corr().
    Continuous×binary → point-biserial.
    """
    pairs = {
        "victim_blaming_x_justification": ("victim_blaming_rulebased", "perp_justified_rulebased"),
        "victim_blaming_x_emo_tone":      ("victim_blaming_rulebased", "tone_emo_ratio_spacy"),
        "justification_x_sens_tone":      ("perp_justified_rulebased", "tone_sens_ratio_spacy"),
        "emo_ratio_x_sens_ratio":         ("tone_emo_ratio_spacy",     "tone_sens_ratio_spacy"),
    }
    return {
        name: float(df[a].corr(df[b]))
        for name, (a, b) in pairs.items()
    }


def compute_method_agreement(df: pd.DataFrame) -> dict:
    """Inter-method agreement (raw rate + Cohen's κ) for VB, PJ, tone."""

    def _dim(cols: tuple) -> dict:
        a, b, c = [df[col] for col in cols]
        return {
            "regex_spacy":  _agreement_pair(a, b),
            "spacy_stanza": _agreement_pair(b, c),
            "regex_stanza": _agreement_pair(a, c),
            "all_three": {
                "rate":  float(((a == b) & (b == c)).mean()),
                "kappa": None,
            },
        }

    return {
        "victim_blaming": _dim(("victim_blaming_regex", "victim_blaming_spacy", "victim_blaming_stanza")),
        "justification":  _dim(("perp_justified_regex", "perp_justified_spacy", "perp_justified_stanza")),
        "tone":           _dim(("tone_regex",            "tone_spacy",           "tone_stanza")),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _save_stats_csv(stats: dict, output_dir: Path) -> None:
    rows = []
    for dim, metrics in stats.items():
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append({"dimension": dim, "metric": f"{metric}_{sub_key}", "value": sub_value})
            else:
                rows.append({"dimension": dim, "metric": metric, "value": value})
    path = output_dir / "dimensions_stats.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved → {path}")


def _save_agreement_csv(agreement: dict, output_dir: Path) -> None:
    rows = []
    for dim, pairs in agreement.items():
        for comparison, metrics in pairs.items():
            rows.append({
                "dimension":      dim,
                "comparison":     comparison,
                "agreement_rate": metrics["rate"],
                "cohen_kappa":    metrics["kappa"],
            })
    path = output_dir / "dimensions_agreement.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved → {path}")


def _save_correlations_csv(correlations: dict, output_dir: Path) -> None:
    path = output_dir / "correlations.csv"
    pd.DataFrame([
        {"pair": k, "correlation": v} for k, v in correlations.items()
    ]).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved → {path}")


def _save_summary_txt(analysis: dict, path: Path) -> None:
    sep  = "=" * 70
    sep2 = "-" * 70
    meta = analysis["metadata"]
    ds   = analysis["stats"]
    corr = analysis["correlations"]
    agr  = analysis["agreement"]

    def _kstr(k):
        return f"{k:.3f}" if k is not None else "N/A"

    lines = [
        sep, "FRAMING DIMENSIONS — DESCRIPTIVE ANALYSIS", sep,
        f"Generated : {meta['generated_at']}",
        f"Input     : {meta['input_file']}",
        f"Articles  : {meta['total_articles']}",
        "",
        sep2, "VICTIM-BLAMING", sep2,
    ]
    vb = ds["victim_blaming"]
    lines += [
        f"Positive : {vb['total_positive']} ({vb['rate']:.1%})",
        f"Ratio    : {vb['ratio_mean']:.3f} ± {vb['ratio_std']:.3f}  "
        f"[{vb['ratio_min']:.3f}, {vb['ratio_max']:.3f}]  median={vb['ratio_median']:.3f}",
        "",
        sep2, "PERPETRATOR JUSTIFICATION", sep2,
    ]
    pj = ds["justification"]
    lines += [
        f"Positive : {pj['total_positive']} ({pj['rate']:.1%})",
        f"Ratio    : {pj['ratio_mean']:.3f} ± {pj['ratio_std']:.3f}  "
        f"[{pj['ratio_min']:.3f}, {pj['ratio_max']:.3f}]  median={pj['ratio_median']:.3f}",
        "",
        sep2, "TONE", sep2,
    ]
    tone = ds["tone"]
    for lbl in ["neutro", "emotivo", "sensazionalistico"]:
        cnt = tone["counts"].get(lbl, 0)
        pct = tone["distribution"].get(lbl, 0.0)
        lines.append(f"  {lbl:<20}: {cnt} ({pct:.1%})")
    lines += [
        f"Emotive ratio        : {tone['emo_ratio_mean']:.3f} ± {tone['emo_ratio_std']:.3f}",
        f"Sensationalistic ratio: {tone['sens_ratio_mean']:.3f} ± {tone['sens_ratio_std']:.3f}",
        "",
        sep2, "CORRELATIONS", sep2,
    ]
    for pair, val in corr.items():
        lines.append(f"  {pair:<45}: {val:+.3f}")
    lines += ["", sep2, "INTER-METHOD AGREEMENT", sep2]
    for dim, pairs in agr.items():
        lines.append(f"  {dim.upper()}")
        for comparison, metrics in pairs.items():
            lines.append(f"    {comparison:<14}: rate={metrics['rate']:.1%}  κ={_kstr(metrics['kappa'])}")
        lines.append("")
    lines += [sep, "DONE", sep]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    if not INPUT.exists():
        logger.critical(f"Input file not found: {INPUT}")
        sys.exit(1)

    logger.info(f"Loading: {INPUT.name}")
    df = pd.read_csv(INPUT, keep_default_na=False)
    logger.info(f"Loaded {len(df)} articles")

    missing = set(_REQUIRED_COLS) - set(df.columns)
    if missing:
        logger.critical(f"Missing required columns: {missing}")
        sys.exit(1)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stats       = compute_dimension_stats(df)
    correlations = compute_correlations(df)
    agreement   = compute_method_agreement(df)

    analysis = {
        "metadata": {
            "total_articles": len(df),
            "generated_at":   generated_at,
            "input_file":     str(INPUT),
        },
        "stats":        stats,
        "correlations": correlations,
        "agreement":    agreement,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(analysis,                      OUTPUT_DIR / "dimensions.json")
    _save_stats_csv(stats,                    OUTPUT_DIR)
    _save_agreement_csv(agreement,            OUTPUT_DIR)
    _save_correlations_csv(correlations,      OUTPUT_DIR)
    _save_summary_txt(analysis,               OUTPUT_DIR / "summary.txt")

    print(
        f"\n✓  Framing dimensions analysis complete  —  {len(df)} articles\n"
        f"   VB rate : {stats['victim_blaming']['rate']:.1%}\n"
        f"   PJ rate : {stats['justification']['rate']:.1%}\n"
        f"   Output  : {OUTPUT_DIR}\n"
    )


if __name__ == "__main__":
    main()