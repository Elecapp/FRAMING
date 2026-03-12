# src/framing_rule_based/justification.py
"""
Perpetrator justification detection using rule-based methods.
"""

import numpy as np
import pandas as pd
from .scoring import sentence_score_justification
from .parsing import (
    select_regex_sentences,
    select_spacy_sentences,
    select_stanza_sentences,
    majority_vote_binary,
)


# ==================== PREDICTION FUNCTIONS ====================

def predict_justification_regex(
        regex_views,
        char_limit,
        max_sentences,
        sent_threshold,
        article_ratio,
):
    """
    Predict perpetrator justification using regex sentence splitting.

    Args:
        regex_views: List of regex-split sentences per article
        char_limit: Max characters to consider
        max_sentences: Max sentences to consider
        sent_threshold: Score threshold for sentence to be positive
        article_ratio: Min ratio of positive sentences for article to be positive

    Returns:
        Tuple of (ratios, labels) as numpy arrays
    """
    ratios = []
    labels = []

    for sentences in regex_views:
        selected = select_regex_sentences(sentences, char_limit, max_sentences)

        if not selected:
            ratios.append(0.0)
            labels.append(0)
            continue

        scores = [sentence_score_justification(s) for s in selected]
        positive_count = sum(1 for sc in scores if sc >= sent_threshold)
        ratio = positive_count / len(selected)

        ratios.append(ratio)
        labels.append(1 if ratio >= article_ratio else 0)

    return np.array(ratios), np.array(labels)


def predict_justification_spacy(
        spacy_views,
        char_limit,
        max_sentences,
        sent_threshold,
        article_ratio,
):
    """Predict perpetrator justification using spaCy parsing."""
    ratios = []
    labels = []

    for sentences in spacy_views:
        selected = select_spacy_sentences(sentences, char_limit, max_sentences)

        if not selected:
            ratios.append(0.0)
            labels.append(0)
            continue

        scores = [sentence_score_justification(s) for s in selected]
        positive_count = sum(1 for sc in scores if sc >= sent_threshold)
        ratio = positive_count / len(selected)

        ratios.append(ratio)
        labels.append(1 if ratio >= article_ratio else 0)

    return np.array(ratios), np.array(labels)


def predict_justification_stanza(
        stanza_views,
        char_limit,
        max_sentences,
        sent_threshold,
        article_ratio,
):
    """Predict perpetrator justification using Stanza parsing."""
    ratios = []
    labels = []

    for sentences in stanza_views:
        selected = select_stanza_sentences(sentences, char_limit, max_sentences)

        if not selected:
            ratios.append(0.0)
            labels.append(0)
            continue

        scores = [sentence_score_justification(s) for s in selected]
        positive_count = sum(1 for sc in scores if sc >= sent_threshold)
        ratio = positive_count / len(selected)

        ratios.append(ratio)
        labels.append(1 if ratio >= article_ratio else 0)

    return np.array(ratios), np.array(labels)


# ==================== APPLY TO DATAFRAME ====================

def apply_justification(df, views, params_dir):
    """
    Apply perpetrator justification detection to dataframe using all three methods.

    Args:
        df: Input dataframe
        views: Dictionary with 'regex', 'spacy', 'stanza' views
        params_dir: Path to directory containing parameter JSON files

    Returns:
        DataFrame with added columns:
        - just_ratio_regex
        - just_ratio_spacy
        - just_ratio_stanza
        - perp_justified_regex
        - perp_justified_spacy
        - perp_justified_stanza
        - perp_justified_rulebased (consensus)
    """
    import json
    from pathlib import Path

    params_dir = Path(params_dir)

    # Load parameters for each method
    methods = {
        "spacy": "chosen_params_just_spacy.json",
        "stanza": "chosen_params_just_stanza.json",
        "regex": "chosen_params_just_regex.json",
    }

    params = {}
    for method, filename in methods.items():
        param_file = params_dir / filename
        if not param_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")

        with open(param_file, "r", encoding="utf-8") as f:
            params[method] = json.load(f)

    # Predict with each method
    print("[JUSTIFICATION] Applying regex method...")
    ratios_regex, labels_regex = predict_justification_regex(
        views["regex"],
        params["regex"]["char_limit"],
        params["regex"]["max_sentences"],
        params["regex"]["sent_threshold"],
        params["regex"]["article_ratio"],
    )

    print("[JUSTIFICATION] Applying spaCy method...")
    ratios_spacy, labels_spacy = predict_justification_spacy(
        views["spacy"],
        params["spacy"]["char_limit"],
        params["spacy"]["max_sentences"],
        params["spacy"]["sent_threshold"],
        params["spacy"]["article_ratio"],
    )

    print("[JUSTIFICATION] Applying Stanza method...")
    ratios_stanza, labels_stanza = predict_justification_stanza(
        views["stanza"],
        params["stanza"]["char_limit"],
        params["stanza"]["max_sentences"],
        params["stanza"]["sent_threshold"],
        params["stanza"]["article_ratio"],
    )

    # Add individual method columns
    df["just_ratio_regex"] = ratios_regex
    df["perp_justified_regex"] = labels_regex

    df["just_ratio_spacy"] = ratios_spacy
    df["perp_justified_spacy"] = labels_spacy

    df["just_ratio_stanza"] = ratios_stanza
    df["perp_justified_stanza"] = labels_stanza

    # Consensus: majority vote (2/3)
    print("[JUSTIFICATION] Computing consensus...")
    consensus = []
    for i in range(len(df)):
        vote = majority_vote_binary(
            labels_regex[i],
            labels_spacy[i],
            labels_stanza[i],
        )
        consensus.append(vote)

    df["perp_justified_rulebased"] = consensus

    print(f"[JUSTIFICATION] Done. Positive rate: {np.mean(consensus):.2%}")

    return df