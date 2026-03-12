# src/framing_rule_based/tone.py
"""
Tone detection (emotive vs sensationalistic) using rule-based methods.
"""

import numpy as np
import pandas as pd
from .scoring import sentence_score_tone_emotive, sentence_score_tone_sensational
from .parsing import (
    select_regex_sentences,
    select_spacy_sentences,
    select_stanza_sentences,
    consensus_multiclass,
)


# ==================== PREDICTION FUNCTIONS ====================

def predict_tone_regex(
        regex_views,
        char_limit,
        max_sentences,
        emo_threshold,
        sens_threshold,
):
    """
    Predict tone using regex sentence splitting.

    Args:
        regex_views: List of regex-split sentences per article
        char_limit: Max characters to consider
        max_sentences: Max sentences to consider
        emo_threshold: Score threshold for emotive tone
        sens_threshold: Score threshold for sensationalistic tone

    Returns:
        Tuple of (emo_ratios, sens_ratios, labels) as numpy arrays
        Labels: "emotivo", "sensazionalistico", or "neutro"
    """
    emo_ratios = []
    sens_ratios = []
    labels = []

    for sentences in regex_views:
        selected = select_regex_sentences(sentences, char_limit, max_sentences)

        if not selected:
            emo_ratios.append(0.0)
            sens_ratios.append(0.0)
            labels.append("neutro")
            continue

        # Score sentences for both dimensions
        emo_scores = [sentence_score_tone_emotive(s) for s in selected]
        sens_scores = [sentence_score_tone_sensational(s) for s in selected]

        # Calculate ratios
        emo_count = sum(1 for sc in emo_scores if sc >= emo_threshold)
        sens_count = sum(1 for sc in sens_scores if sc >= sens_threshold)

        emo_ratio = emo_count / len(selected)
        sens_ratio = sens_count / len(selected)

        emo_ratios.append(emo_ratio)
        sens_ratios.append(sens_ratio)

        # Determine label (emotive takes priority if both)
        if emo_ratio > 0:
            labels.append("emotivo")
        elif sens_ratio > 0:
            labels.append("sensazionalistico")
        else:
            labels.append("neutro")

    return np.array(emo_ratios), np.array(sens_ratios), np.array(labels)


def predict_tone_spacy(
        spacy_views,
        char_limit,
        max_sentences,
        emo_threshold,
        sens_threshold,
):
    """Predict tone using spaCy parsing."""
    emo_ratios = []
    sens_ratios = []
    labels = []

    for sentences in spacy_views:
        selected = select_spacy_sentences(sentences, char_limit, max_sentences)

        if not selected:
            emo_ratios.append(0.0)
            sens_ratios.append(0.0)
            labels.append("neutro")
            continue

        emo_scores = [sentence_score_tone_emotive(s) for s in selected]
        sens_scores = [sentence_score_tone_sensational(s) for s in selected]

        emo_count = sum(1 for sc in emo_scores if sc >= emo_threshold)
        sens_count = sum(1 for sc in sens_scores if sc >= sens_threshold)

        emo_ratio = emo_count / len(selected)
        sens_ratio = sens_count / len(selected)

        emo_ratios.append(emo_ratio)
        sens_ratios.append(sens_ratio)

        if emo_ratio > 0:
            labels.append("emotivo")
        elif sens_ratio > 0:
            labels.append("sensazionalistico")
        else:
            labels.append("neutro")

    return np.array(emo_ratios), np.array(sens_ratios), np.array(labels)


def predict_tone_stanza(
        stanza_views,
        char_limit,
        max_sentences,
        emo_threshold,
        sens_threshold,
):
    """Predict tone using Stanza parsing."""
    emo_ratios = []
    sens_ratios = []
    labels = []

    for sentences in stanza_views:
        selected = select_stanza_sentences(sentences, char_limit, max_sentences)

        if not selected:
            emo_ratios.append(0.0)
            sens_ratios.append(0.0)
            labels.append("neutro")
            continue

        emo_scores = [sentence_score_tone_emotive(s) for s in selected]
        sens_scores = [sentence_score_tone_sensational(s) for s in selected]

        emo_count = sum(1 for sc in emo_scores if sc >= emo_threshold)
        sens_count = sum(1 for sc in sens_scores if sc >= sens_threshold)

        emo_ratio = emo_count / len(selected)
        sens_ratio = sens_count / len(selected)

        emo_ratios.append(emo_ratio)
        sens_ratios.append(sens_ratio)

        if emo_ratio > 0:
            labels.append("emotivo")
        elif sens_ratio > 0:
            labels.append("sensazionalistico")
        else:
            labels.append("neutro")

    return np.array(emo_ratios), np.array(sens_ratios), np.array(labels)


# ==================== APPLY TO DATAFRAME ====================

def apply_tone(df, views, params_dir):
    """
    Apply tone detection to dataframe using all three methods.

    Args:
        df: Input dataframe
        views: Dictionary with 'regex', 'spacy', 'stanza' views
        params_dir: Path to directory containing parameter JSON files

    Returns:
        DataFrame with added columns:
        - tone_emo_ratio_regex
        - tone_emo_ratio_spacy
        - tone_emo_ratio_stanza
        - tone_sens_ratio_regex
        - tone_sens_ratio_spacy
        - tone_sens_ratio_stanza
        - tone_regex
        - tone_spacy
        - tone_stanza
        - tone_rulebased (consensus)
    """
    import json
    from pathlib import Path

    params_dir = Path(params_dir)

    # Load parameters for each method
    methods = {
        "spacy": "chosen_params_tone_spacy.json",
        "stanza": "chosen_params_tone_stanza.json",
        "regex": "chosen_params_tone_regex.json",
    }

    params = {}
    for method, filename in methods.items():
        param_file = params_dir / filename
        if not param_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")

        with open(param_file, "r", encoding="utf-8") as f:
            params[method] = json.load(f)

    # Predict with each method
    print("[TONE] Applying regex method...")
    emo_ratios_regex, sens_ratios_regex, labels_regex = predict_tone_regex(
        views["regex"],
        params["regex"]["char_limit"],
        params["regex"]["max_sentences"],
        params["regex"]["emo_threshold"],
        params["regex"]["sens_threshold"],
    )

    print("[TONE] Applying spaCy method...")
    emo_ratios_spacy, sens_ratios_spacy, labels_spacy = predict_tone_spacy(
        views["spacy"],
        params["spacy"]["char_limit"],
        params["spacy"]["max_sentences"],
        params["spacy"]["emo_threshold"],
        params["spacy"]["sens_threshold"],
    )

    print("[TONE] Applying Stanza method...")
    emo_ratios_stanza, sens_ratios_stanza, labels_stanza = predict_tone_stanza(
        views["stanza"],
        params["stanza"]["char_limit"],
        params["stanza"]["max_sentences"],
        params["stanza"]["emo_threshold"],
        params["stanza"]["sens_threshold"],
    )

    # Add individual method columns
    df["tone_emo_ratio_regex"] = emo_ratios_regex
    df["tone_sens_ratio_regex"] = sens_ratios_regex
    df["tone_regex"] = labels_regex

    df["tone_emo_ratio_spacy"] = emo_ratios_spacy
    df["tone_sens_ratio_spacy"] = sens_ratios_spacy
    df["tone_spacy"] = labels_spacy

    df["tone_emo_ratio_stanza"] = emo_ratios_stanza
    df["tone_sens_ratio_stanza"] = sens_ratios_stanza
    df["tone_stanza"] = labels_stanza

    # Consensus: multiclass with tie-breaking
    print("[TONE] Computing consensus...")
    consensus = []
    for i in range(len(df)):
        vote = consensus_multiclass([
            labels_regex[i],
            labels_spacy[i],
            labels_stanza[i],
        ])
        consensus.append(vote)

    df["tone_rulebased"] = consensus

    # Distribution stats
    from collections import Counter
    dist = Counter(consensus)
    print(f"[TONE] Done. Distribution:")
    for label in ["emotivo", "sensazionalistico", "neutro"]:
        count = dist.get(label, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    return df