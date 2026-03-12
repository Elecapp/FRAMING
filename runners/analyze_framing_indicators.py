# runners/analyze_framing_indicators.py
# Descriptive analysis of syntactic framing indicators (victim agency,
# narrative focus) on unseen corpora.
#
# Intended for: LOC_v2_event_detection_framing.csv or similar unannotated data.
# These indicators have no ground truth — analysis is purely descriptive.
#
# Outputs (reports/framing_indicators/):
#   indicators.json       — full results
#   indicators_stats.csv  — flat stats table
#   correlations.csv      — correlations between agency, focus, and dimensions
#   summary.txt           — human-readable report

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

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

INPUT = ROOT / "data" / "processed" / "LOC_v2_event_detection_framing.csv"
OUTPUT_DIR = ROOT / "reports" / "framing_indicators"

# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

_REQUIRED_COLS: list[str] = [
    "victim_agency_class",
    "victim_agency_idx",
    "victim_passive_ratio",
    "victim_object_ratio",
    "victim_role_entropy",
    "victim_subj_active",
    "victim_subj_passive",
    "victim_object",
    "focus_rulebased",
    "focus_victim_score",
    "focus_perp_score",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_indicator_stats(df: pd.DataFrame) -> dict:
    """Descriptive statistics for victim agency and narrative focus."""

    def _class(col: str) -> dict:
        vc = df[col].value_counts()
        vc_norm = df[col].value_counts(normalize=True)
        return {
            "counts": {k: int(v) for k, v in vc.items()},
            "distribution": {k: float(v) for k, v in vc_norm.items()},
        }

    def _continuous(col: str) -> dict:
        s = df[col]
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    return {
        "victim_agency": {
            **_class("victim_agency_class"),
            "agency_idx": _continuous("victim_agency_idx"),
            "passive_ratio": _continuous("victim_passive_ratio"),
            "object_ratio": _continuous("victim_object_ratio"),
            "role_entropy": _continuous("victim_role_entropy"),
            "counts_raw": {
                "subj_active_mean": float(df["victim_subj_active"].mean()),
                "subj_passive_mean": float(df["victim_subj_passive"].mean()),
                "object_mean": float(df["victim_object"].mean()),
            },
        },
        "narrative_focus": {
            **_class("focus_rulebased"),
            "victim_score": _continuous("focus_victim_score"),
            "perp_score": _continuous("focus_perp_score"),
        },
    }


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Correlations between syntactic indicators and framing dimensions.
    Only computed when framing dimension columns are present in the dataset.
    """
    pairs = {
        "agency_idx_x_victim_score": ("victim_agency_idx", "focus_victim_score"),
        "agency_idx_x_perp_score": ("victim_agency_idx", "focus_perp_score"),
        "passive_ratio_x_perp_score": ("victim_passive_ratio", "focus_perp_score"),
        "entropy_x_agency_idx": ("victim_role_entropy", "victim_agency_idx"),
    }

    # Add cross-dimension correlations if framing columns are available
    optional_pairs = {
        "agency_idx_x_victim_blaming": ("victim_agency_idx", "victim_blaming_rulebased"),
        "agency_idx_x_justification": ("victim_agency_idx", "perp_justified_rulebased"),
        "victim_focus_x_emo_tone": ("focus_victim_score", "tone_emo_ratio_spacy"),
        "perp_focus_x_sens_tone": ("focus_perp_score", "tone_sens_ratio_spacy"),
    }
    for name, (a, b) in optional_pairs.items():
        if a in df.columns and b in df.columns:
            pairs[name] = (a, b)

    return {
        name: float(df[a].corr(df[b]))
        for name, (a, b) in pairs.items()
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
                    if isinstance(sub_value, dict):
                        for k2, v2 in sub_value.items():
                            rows.append({"dimension": dim, "metric": f"{metric}_{sub_key}_{k2}", "value": v2})
                    else:
                        rows.append({"dimension": dim, "metric": f"{metric}_{sub_key}", "value": sub_value})
            else:
                rows.append({"dimension": dim, "metric": metric, "value": value})
    path = output_dir / "indicators_stats.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved → {path}")


def _save_correlations_csv(correlations: dict, output_dir: Path) -> None:
    path = output_dir / "correlations.csv"
    pd.DataFrame([
        {"pair": k, "correlation": v} for k, v in correlations.items()
    ]).to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved → {path}")


def _save_summary_txt(analysis: dict, path: Path) -> None:
    sep = "=" * 70
    sep2 = "-" * 70
    meta = analysis["metadata"]
    ds = analysis["stats"]
    corr = analysis["correlations"]

    def _cfmt(d: dict) -> str:
        return (f"mean={d['mean']:.3f} ± {d['std']:.3f}  "
                f"[{d['min']:.3f}, {d['max']:.3f}]  median={d['median']:.3f}")

    lines = [
        sep, "FRAMING INDICATORS — DESCRIPTIVE ANALYSIS", sep,
        f"Generated : {meta['generated_at']}",
        f"Input     : {meta['input_file']}",
        f"Articles  : {meta['total_articles']}",
        "",
        sep2, "VICTIM AGENCY", sep2,
    ]
    ag = ds["victim_agency"]
    for lbl in ["alta", "media", "bassa"]:
        cnt = ag["counts"].get(lbl, 0)
        pct = ag["distribution"].get(lbl, 0.0)
        lines.append(f"  {lbl:<6}: {cnt} ({pct:.1%})")
    lines += [
        f"Agency index  : {_cfmt(ag['agency_idx'])}",
        f"Passive ratio : {_cfmt(ag['passive_ratio'])}",
        f"Object ratio  : {_cfmt(ag['object_ratio'])}",
        f"Role entropy  : {_cfmt(ag['role_entropy'])}",
        f"Avg roles     : subj_active={ag['counts_raw']['subj_active_mean']:.2f}  "
        f"subj_passive={ag['counts_raw']['subj_passive_mean']:.2f}  "
        f"object={ag['counts_raw']['object_mean']:.2f}",
        "",
        sep2, "NARRATIVE FOCUS", sep2,
    ]
    fc = ds["narrative_focus"]
    for lbl in ["vittima", "carnefice", "bilanciato"]:
        cnt = fc["counts"].get(lbl, 0)
        pct = fc["distribution"].get(lbl, 0.0)
        lines.append(f"  {lbl:<12}: {cnt} ({pct:.1%})")
    lines += [
        f"Victim score : {_cfmt(fc['victim_score'])}",
        f"Perp score   : {_cfmt(fc['perp_score'])}",
        "",
        sep2, "CORRELATIONS", sep2,
    ]
    for pair, val in corr.items():
        lines.append(f"  {pair:<45}: {val:+.3f}")
    lines += ["", sep, "DONE", sep]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved : {path}")


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

    stats = compute_indicator_stats(df)
    correlations = compute_correlations(df)

    analysis = {
        "metadata": {
            "total_articles": len(df),
            "generated_at": generated_at,
            "input_file": str(INPUT),
        },
        "stats": stats,
        "correlations": correlations,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(analysis, OUTPUT_DIR / "indicators.json")
    _save_stats_csv(stats, OUTPUT_DIR)
    _save_correlations_csv(correlations, OUTPUT_DIR)
    _save_summary_txt(analysis, OUTPUT_DIR / "summary.txt")

    ag = stats["victim_agency"]
    fc = stats["narrative_focus"]
    print(
        f"\n-  Framing indicators analysis complete  —  {len(df)} articles\n"
        f"   Agency alta/media/bassa : "
        f"{ag['distribution'].get('alta', 0):.1%} / "
        f"{ag['distribution'].get('media', 0):.1%} / "
        f"{ag['distribution'].get('bassa', 0):.1%}\n"
        f"   Focus vittima/carnefice/bilanciato : "
        f"{fc['distribution'].get('vittima', 0):.1%} / "
        f"{fc['distribution'].get('carnefice', 0):.1%} / "
        f"{fc['distribution'].get('bilanciato', 0):.1%}\n"
        f"   Output : {OUTPUT_DIR}\n"
    )


if __name__ == "__main__":
    main()
