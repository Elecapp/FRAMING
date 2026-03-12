# runners/run_rule_based_param_tuning.py
# Grid search over aggregation parameters for rule-based framing detectors.
#
# The lexicons are fixed (src/framing_rule_based/lexicons.py).
# The only parameters estimated here are the aggregation hyperparameters:
#
#   char_limit      — max cumulative characters read per article
#   max_sentences   — max sentences read per article
#   sent_threshold  — min sentence score to count as positive
#   article_ratio   — min fraction of positive sentences for article label
#
# For tone, emo_threshold and sens_threshold are tuned separately because
# the score distributions for emotive and sensationalistic signals differ.
#
# Workflow
# --------
#   1. Load FAS ground truth (preprocessed).
#   2. Build linguistic views once (regex / spaCy / Stanza).
#   3. Grid search each detector × method; select by F1 on FAS.
#   4. Save chosen params to config/framing_params/*.json.
#   5. Apply detectors with chosen params and evaluate.
#   6. Save predictions, per-method metrics, and a summary report.

from __future__ import annotations

import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

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
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "FAS_preprocessed.csv"
PARAMS_DIR = ROOT / "config" / "framing_params"
REPORT_DIR = ROOT / "reports" / "rule_based_tuning"
OUT_PATH = ROOT / "data" / "processed" / "FAS_rule_based_predictions.csv"

# ---------------------------------------------------------------------------
# Imports from package (after ROOT is set)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT))

from src.framing_rule_based.parsing import build_rule_based_views
from src.framing_rule_based.victim_blaming import (
    predict_victim_blaming_regex,
    predict_victim_blaming_spacy,
    predict_victim_blaming_stanza,
)
from src.framing_rule_based.justification import (
    predict_justification_regex,
    predict_justification_spacy,
    predict_justification_stanza,
)
from src.framing_rule_based.tone import (
    predict_tone_regex,
    predict_tone_spacy,
    predict_tone_stanza,
)

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# Shared grid for all binary detectors (VB, PJ) and tone components
_CHAR_LIMITS = [1000, 1500, 2000, 3000, 4000]
_MAX_SENTENCES = [5, 10, 15, 20, 40]
_SENT_THRESHOLDS = [0.5, 0.7, 0.8, 1.0, 1.2]
_ARTICLE_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.4]


def _binary_grid() -> list[dict[str, Any]]:
    """All combinations of binary aggregation hyperparameters."""
    return [
        {
            "char_limit": cl,
            "max_sentences": ms,
            "sent_threshold": st,
            "article_ratio": ar,
        }
        for cl, ms, st, ar in product(
            _CHAR_LIMITS, _MAX_SENTENCES, _SENT_THRESHOLDS, _ARTICLE_RATIOS
        )
    ]


def _tone_grid() -> list[dict[str, Any]]:
    """
    Grid for a single tone component (emotive or sensationalistic).
    Uses the same four axes as the binary grid.
    """
    return _binary_grid()


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _eval_binary(y_true: list, y_pred: list) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _eval_multiclass(y_true: list, y_pred: list) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _best_by_f1(results: list[dict]) -> dict:
    """Return the result dict with the highest F1."""
    return max(results, key=lambda x: x["metrics"]["f1"])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Binary detector tuning
# ---------------------------------------------------------------------------

def _tune_binary(
        predict_fn,
        views: list,
        y_true: list[int],
        desc: str,
) -> tuple[dict, dict, list[dict]]:
    """
    Grid search over binary aggregation parameters.

    Parameters
    ----------
    predict_fn : callable
        One of predict_victim_blaming_*/predict_justification_* functions.
        Signature: (views, char_limit, max_sentences, sent_threshold,
                    article_ratio) → (ratios, labels).
    views : list
        Parsed sentence lists for all articles (regex / spaCy / Stanza).
    y_true : list[int]
        Ground-truth binary labels from FAS.
    desc : str
        Label for the tqdm progress bar.

    Returns
    -------
    best_params : dict
    best_metrics : dict
    all_results : list[dict]
    """
    grid = _binary_grid()
    results = []

    for params in tqdm(grid, desc=desc, leave=False):
        _, y_pred = predict_fn(
            views,
            params["char_limit"],
            params["max_sentences"],
            params["sent_threshold"],
            params["article_ratio"],
        )
        metrics = _eval_binary(y_true, y_pred.tolist())
        results.append({"params": params, "metrics": metrics})

    best = _best_by_f1(results)
    return best["params"], best["metrics"], results


# ---------------------------------------------------------------------------
# Tone component tuning
# ---------------------------------------------------------------------------

def _tune_tone_component(
        predict_fn,
        views: list,
        y_true_binary: list[int],
        component: str,
        desc: str,
) -> tuple[dict, dict, list[dict]]:
    """
    Grid search over a single tone component (emotive OR sensationalistic).

    The non-tuned component is fixed with a neutral (permissive) config so
    it never fires, isolating the component being tuned.

    Parameters
    ----------
    predict_fn : callable
        One of predict_tone_regex/spacy/stanza.
        Signature: (views, char_limit, max_sentences, emo_threshold,
                    sens_threshold) → (emo_ratios, sens_ratios, labels).
    views : list
    y_true_binary : list[int]
        1 where the article has the target tone, 0 otherwise.
    component : str
        ``'emo'`` or ``'sens'``.
    desc : str

    Returns
    -------
    best_params : dict
    best_metrics : dict
    all_results : list[dict]
    """
    # Fixed params for the component NOT being tuned: threshold so high it
    # never fires, so the other component can be isolated.
    _NEVER_FIRE = {
        "char_limit": 4000,
        "max_sentences": 40,
        "emo_threshold": 999.0,
        "sens_threshold": 999.0,
    }

    grid = _tone_grid()
    results = []

    for params in tqdm(grid, desc=desc, leave=False):
        if component == "emo":
            _, _, labels = predict_fn(
                views,
                params["char_limit"],
                params["max_sentences"],
                emo_threshold=params["sent_threshold"],
                sens_threshold=_NEVER_FIRE["sens_threshold"],
            )
            y_pred = [1 if lbl == "emotivo" else 0 for lbl in labels]
        elif component == "sens":
            _, _, labels = predict_fn(
                views,
                params["char_limit"],
                params["max_sentences"],
                emo_threshold=_NEVER_FIRE["emo_threshold"],
                sens_threshold=params["sent_threshold"],
            )
            y_pred = [1 if lbl == "sensazionalistico" else 0 for lbl in labels]
        else:
            raise ValueError(f"Unknown component: {component!r}")

        metrics = _eval_binary(y_true_binary, y_pred)
        results.append({"params": params, "metrics": metrics})

    best = _best_by_f1(results)
    return best["params"], best["metrics"], results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load FAS ground truth
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        logger.critical(f"FAS preprocessed file not found: {DATA_PATH}")
        sys.exit(1)

    logger.info(f"Loading FAS: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, keep_default_na=False)

    required = ["text_linguistic", "victim_blaming_gt", "perp_justified_gt", "tone_label_gt"]
    missing = set(required) - set(df.columns)
    if missing:
        logger.critical(f"Missing columns in FAS: {missing}")
        sys.exit(1)

    y_victim = df["victim_blaming_gt"].astype(int).tolist()
    y_just = df["perp_justified_gt"].astype(int).tolist()
    y_tone = df["tone_label_gt"].tolist()

    y_tone_emo = [1 if t == "emotivo" else 0 for t in y_tone]
    y_tone_sens = [1 if t == "sensazionalistico" else 0 for t in y_tone]

    logger.info(
        f"FAS: {len(df)} articles  |  "
        f"VB={sum(y_victim)}  PJ={sum(y_just)}  "
        f"emo={sum(y_tone_emo)}  sens={sum(y_tone_sens)}"
    )

    # ------------------------------------------------------------------
    # 2. Build linguistic views (once — reused by all detectors)
    # ------------------------------------------------------------------
    logger.info("Building linguistic views (regex / spaCy / Stanza)...")
    views = build_rule_based_views(df["text_linguistic"], show_progress=True)

    # ------------------------------------------------------------------
    # 3. Victim-blaming grid search
    # ------------------------------------------------------------------
    logger.info("Tuning victim-blaming detectors...")

    params_vb_regex, metrics_vb_regex, results_vb_regex = _tune_binary(
        predict_victim_blaming_regex, views["regex"], y_victim, "VB regex"
    )
    params_vb_spacy, metrics_vb_spacy, results_vb_spacy = _tune_binary(
        predict_victim_blaming_spacy, views["spacy"], y_victim, "VB spaCy"
    )
    params_vb_stanza, metrics_vb_stanza, results_vb_stanza = _tune_binary(
        predict_victim_blaming_stanza, views["stanza"], y_victim, "VB Stanza"
    )

    _save_json(results_vb_regex, REPORT_DIR / "victim_blaming_regex_grid.json")
    _save_json(results_vb_spacy, REPORT_DIR / "victim_blaming_spacy_grid.json")
    _save_json(results_vb_stanza, REPORT_DIR / "victim_blaming_stanza_grid.json")

    _save_json(params_vb_regex, PARAMS_DIR / "chosen_params_victim_regex.json")
    _save_json(params_vb_spacy, PARAMS_DIR / "chosen_params_victim_spacy.json")
    _save_json(params_vb_stanza, PARAMS_DIR / "chosen_params_victim_stanza.json")

    logger.info(
        f"VB best — regex F1={metrics_vb_regex['f1']:.3f}  "
        f"spaCy F1={metrics_vb_spacy['f1']:.3f}  "
        f"Stanza F1={metrics_vb_stanza['f1']:.3f}"
    )

    # ------------------------------------------------------------------
    # 4. Perpetrator-justification grid search
    # ------------------------------------------------------------------
    logger.info("Tuning justification detectors...")

    params_just_regex, metrics_just_regex, results_just_regex = _tune_binary(
        predict_justification_regex, views["regex"], y_just, "PJ regex"
    )
    params_just_spacy, metrics_just_spacy, results_just_spacy = _tune_binary(
        predict_justification_spacy, views["spacy"], y_just, "PJ spaCy"
    )
    params_just_stanza, metrics_just_stanza, results_just_stanza = _tune_binary(
        predict_justification_stanza, views["stanza"], y_just, "PJ Stanza"
    )

    _save_json(results_just_regex, REPORT_DIR / "justification_regex_grid.json")
    _save_json(results_just_spacy, REPORT_DIR / "justification_spacy_grid.json")
    _save_json(results_just_stanza, REPORT_DIR / "justification_stanza_grid.json")

    _save_json(params_just_regex, PARAMS_DIR / "chosen_params_just_regex.json")
    _save_json(params_just_spacy, PARAMS_DIR / "chosen_params_just_spacy.json")
    _save_json(params_just_stanza, PARAMS_DIR / "chosen_params_just_stanza.json")

    logger.info(
        f"PJ best — regex F1={metrics_just_regex['f1']:.3f}  "
        f"spaCy F1={metrics_just_spacy['f1']:.3f}  "
        f"Stanza F1={metrics_just_stanza['f1']:.3f}"
    )

    # ------------------------------------------------------------------
    # 5. Tone grid search (emo and sens tuned independently per method)
    # ------------------------------------------------------------------
    logger.info("Tuning tone detectors (emotive)...")

    params_tone_emo_regex, m_te_re, r_te_re = _tune_tone_component(predict_tone_regex, views["regex"], y_tone_emo,
                                                                   "emo", "Tone-emo regex")
    params_tone_emo_spacy, m_te_sp, r_te_sp = _tune_tone_component(predict_tone_spacy, views["spacy"], y_tone_emo,
                                                                   "emo", "Tone-emo spaCy")
    params_tone_emo_stanza, m_te_st, r_te_st = _tune_tone_component(predict_tone_stanza, views["stanza"], y_tone_emo,
                                                                    "emo", "Tone-emo Stanza")

    logger.info("Tuning tone detectors (sensationalistic)...")

    params_tone_sens_regex, m_ts_re, r_ts_re = _tune_tone_component(predict_tone_regex, views["regex"], y_tone_sens,
                                                                    "sens", "Tone-sens regex")
    params_tone_sens_spacy, m_ts_sp, r_ts_sp = _tune_tone_component(predict_tone_spacy, views["spacy"], y_tone_sens,
                                                                    "sens", "Tone-sens spaCy")
    params_tone_sens_stanza, m_ts_st, r_ts_st = _tune_tone_component(predict_tone_stanza, views["stanza"], y_tone_sens,
                                                                     "sens", "Tone-sens Stanza")

    for name, results in [
        ("tone_emo_regex_grid", r_te_re),
        ("tone_emo_spacy_grid", r_te_sp),
        ("tone_emo_stanza_grid", r_te_st),
        ("tone_sens_regex_grid", r_ts_re),
        ("tone_sens_spacy_grid", r_ts_sp),
        ("tone_sens_stanza_grid", r_ts_st),
    ]:
        _save_json(results, REPORT_DIR / f"{name}.json")

    # Save combined tone params per method (emo_threshold + sens_threshold merged)
    for method, p_emo, p_sens in [
        ("regex", params_tone_emo_regex, params_tone_sens_regex),
        ("spacy", params_tone_emo_spacy, params_tone_sens_spacy),
        ("stanza", params_tone_emo_stanza, params_tone_sens_stanza),
    ]:
        combined = {
            "char_limit": p_emo["char_limit"],  # use emo char_limit as base
            "max_sentences": p_emo["max_sentences"],
            "emo_threshold": p_emo["sent_threshold"],
            "sens_threshold": p_sens["sent_threshold"],
        }
        _save_json(combined, PARAMS_DIR / f"chosen_params_tone_{method}.json")

    logger.info(
        f"Tone-emo best  — regex F1={m_te_re['f1']:.3f}  spaCy F1={m_te_sp['f1']:.3f}  Stanza F1={m_te_st['f1']:.3f}\n"
        f"Tone-sens best — regex F1={m_ts_re['f1']:.3f}  spaCy F1={m_ts_sp['f1']:.3f}  Stanza F1={m_ts_st['f1']:.3f}"
    )

    # ------------------------------------------------------------------
    # 6. Apply tuned detectors on full FAS and evaluate
    # ------------------------------------------------------------------
    logger.info("Applying tuned detectors to FAS...")

    from src.framing_rule_based.victim_blaming import apply_victim_blaming
    from src.framing_rule_based.justification import apply_justification
    from src.framing_rule_based.tone import apply_tone
    from src.framing_rule_based.roles_focus import apply_roles_focus

    df_pred = df.copy()
    df_pred = apply_victim_blaming(df_pred, views, PARAMS_DIR)
    df_pred = apply_justification(df_pred, views, PARAMS_DIR)
    df_pred = apply_tone(df_pred, views, PARAMS_DIR)
    df_pred = apply_roles_focus(df_pred, views)

    df_pred.to_csv(OUT_PATH, index=False, encoding="utf-8")
    logger.info(f"Predictions saved → {OUT_PATH}")

    # ------------------------------------------------------------------
    # 7. Per-method evaluation on FAS
    # ------------------------------------------------------------------
    def _col(df, name):
        return df[name].tolist()

    victim_metrics = {
        "regex": _eval_binary(_col(df_pred, "victim_blaming_gt"), _col(df_pred, "victim_blaming_regex")),
        "spacy": _eval_binary(_col(df_pred, "victim_blaming_gt"), _col(df_pred, "victim_blaming_spacy")),
        "stanza": _eval_binary(_col(df_pred, "victim_blaming_gt"), _col(df_pred, "victim_blaming_stanza")),
        "majority_vote": _eval_binary(_col(df_pred, "victim_blaming_gt"), _col(df_pred, "victim_blaming_rulebased")),
    }
    just_metrics = {
        "regex": _eval_binary(_col(df_pred, "perp_justified_gt"), _col(df_pred, "perp_justified_regex")),
        "spacy": _eval_binary(_col(df_pred, "perp_justified_gt"), _col(df_pred, "perp_justified_spacy")),
        "stanza": _eval_binary(_col(df_pred, "perp_justified_gt"), _col(df_pred, "perp_justified_stanza")),
        "majority_vote": _eval_binary(_col(df_pred, "perp_justified_gt"), _col(df_pred, "perp_justified_rulebased")),
    }
    tone_metrics = {
        "regex": _eval_multiclass(_col(df_pred, "tone_label_gt"), _col(df_pred, "tone_regex")),
        "spacy": _eval_multiclass(_col(df_pred, "tone_label_gt"), _col(df_pred, "tone_spacy")),
        "stanza": _eval_multiclass(_col(df_pred, "tone_label_gt"), _col(df_pred, "tone_stanza")),
        "majority_vote": _eval_multiclass(_col(df_pred, "tone_label_gt"), _col(df_pred, "tone_rulebased")),
    }

    _save_json(victim_metrics, REPORT_DIR / "victim_blaming_metrics.json")
    _save_json(just_metrics, REPORT_DIR / "justification_metrics.json")
    _save_json(tone_metrics, REPORT_DIR / "tone_metrics.json")

    # ------------------------------------------------------------------
    # 8. Summary report
    # ------------------------------------------------------------------
    sep = "=" * 70
    sep2 = "-" * 70

    def _fmt_metrics(m: dict[str, dict]) -> list[str]:
        lines = []
        for method, met in m.items():
            lines.append(
                f"  {method:<16} F1={met['f1']:.3f}  "
                f"P={met['precision']:.3f}  R={met['recall']:.3f}  "
                f"Acc={met['accuracy']:.3f}"
            )
        return lines

    summary_lines = [
        sep, "RULE-BASED PARAMETER TUNING — SUMMARY", sep, "",
        "SELECTED PARAMETERS", sep2,
        f"  VB    regex  : {params_vb_regex}",
        f"  VB    spaCy  : {params_vb_spacy}",
        f"  VB    Stanza : {params_vb_stanza}",
        f"  PJ    regex  : {params_just_regex}",
        f"  PJ    spaCy  : {params_just_spacy}",
        f"  PJ    Stanza : {params_just_stanza}",
        f"  Tone-emo  regex  : {params_tone_emo_regex}",
        f"  Tone-emo  spaCy  : {params_tone_emo_spacy}",
        f"  Tone-emo  Stanza : {params_tone_emo_stanza}",
        f"  Tone-sens regex  : {params_tone_sens_regex}",
        f"  Tone-sens spaCy  : {params_tone_sens_spacy}",
        f"  Tone-sens Stanza : {params_tone_sens_stanza}",
        "",
        "VICTIM BLAMING PERFORMANCE", sep2,
        *_fmt_metrics(victim_metrics),
        "",
        "PERPETRATOR JUSTIFICATION PERFORMANCE", sep2,
        *_fmt_metrics(just_metrics),
        "",
        "TONE PERFORMANCE (macro)", sep2,
        *_fmt_metrics(tone_metrics),
        "",
        sep, "TUNING COMPLETED", sep,
    ]

    summary_text = "\n".join(summary_lines)
    (REPORT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")
    logger.info("\n" + summary_text)


if __name__ == "__main__":
    main()
