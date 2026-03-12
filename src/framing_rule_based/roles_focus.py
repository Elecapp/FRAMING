# src/framing_rule_based/roles_focus.py
"""
Victim agency and narrative focus detection using syntactic analysis.
Uses dependency parsing from spaCy (primary) with Stanza fallback.
"""

import numpy as np
import pandas as pd
from collections import Counter
from .lexicons import VICTIM_UNIGRAMS, VICTIM_MULTI, PERP_UNIGRAMS, PERP_MULTI


# ==================== ENTITY DETECTION ====================

def _detect_entities_in_sentence(sent):
    """
    Detect victim and perpetrator mentions in a sentence.

    Args:
        sent: spaCy Span or Stanza Sentence

    Returns:
        Tuple of (victim_mentions, perp_mentions) as lists of dicts
        Each dict contains: {'text': str, 'role': str, 'dep': str}
    """
    victim_mentions = []
    perp_mentions = []

    # spaCy processing
    if hasattr(sent, "doc"):
        text = sent.text.lower()

        # Check multi-word expressions first
        for expr in VICTIM_MULTI:
            if expr in text:
                # Find syntactic role
                for token in sent:
                    if expr in token.text.lower() or token.lemma_.lower() in VICTIM_UNIGRAMS:
                        victim_mentions.append({
                            'text': expr,
                            'role': 'victim',
                            'dep': token.dep_,
                        })
                        break

        for expr in PERP_MULTI:
            if expr in text:
                for token in sent:
                    if expr in token.text.lower() or token.lemma_.lower() in PERP_UNIGRAMS:
                        perp_mentions.append({
                            'text': expr,
                            'role': 'perp',
                            'dep': token.dep_,
                        })
                        break

        # Check unigrams
        for token in sent:
            lemma = token.lemma_.lower()

            if lemma in VICTIM_UNIGRAMS:
                # Skip if already captured in multi-word
                if not any(m['dep'] == token.dep_ for m in victim_mentions):
                    victim_mentions.append({
                        'text': token.text,
                        'role': 'victim',
                        'dep': token.dep_,
                    })

            if lemma in PERP_UNIGRAMS:
                if not any(m['dep'] == token.dep_ for m in perp_mentions):
                    perp_mentions.append({
                        'text': token.text,
                        'role': 'perp',
                        'dep': token.dep_,
                    })

    # Stanza processing
    elif hasattr(sent, "words"):
        text = sent.text.lower()

        # Multi-word expressions
        for expr in VICTIM_MULTI:
            if expr in text:
                for word in sent.words:
                    if expr in word.text.lower() or word.lemma.lower() in VICTIM_UNIGRAMS:
                        victim_mentions.append({
                            'text': expr,
                            'role': 'victim',
                            'dep': word.deprel,
                        })
                        break

        for expr in PERP_MULTI:
            if expr in text:
                for word in sent.words:
                    if expr in word.text.lower() or word.lemma.lower() in PERP_UNIGRAMS:
                        perp_mentions.append({
                            'text': expr,
                            'role': 'perp',
                            'dep': word.deprel,
                        })
                        break

        # Unigrams
        for word in sent.words:
            lemma = word.lemma.lower()

            if lemma in VICTIM_UNIGRAMS:
                if not any(m['dep'] == word.deprel for m in victim_mentions):
                    victim_mentions.append({
                        'text': word.text,
                        'role': 'victim',
                        'dep': word.deprel,
                    })

            if lemma in PERP_UNIGRAMS:
                if not any(m['dep'] == word.deprel for m in perp_mentions):
                    perp_mentions.append({
                        'text': word.text,
                        'role': 'perp',
                        'dep': word.deprel,
                    })

    return victim_mentions, perp_mentions


# ==================== ROLE CLASSIFICATION ====================

def _classify_role(dep):
    """
    Classify dependency relation into syntactic role.

    Args:
        dep: Dependency relation string (spaCy or Stanza format)

    Returns:
        'subj_active', 'subj_passive', 'object', or None
    """
    dep = dep.lower()

    # Active subject
    if dep in {'nsubj', 'csubj'}:
        return 'subj_active'

    # Passive subject
    if dep in {'nsubj:pass', 'nsubjpass', 'csubj:pass'}:
        return 'subj_passive'

    # Object
    if dep in {'obj', 'dobj', 'iobj', 'obl'}:
        return 'object'

    return None

# ==================== VICTIM AGENCY ====================


def compute_victim_agency(spacy_views):
    """
    Compute victim agency metrics based on syntactic roles.

    Args:
        spacy_views: List of spaCy sentence lists per article

    Returns:
        DataFrame with columns:
        - victim_subj_active: Count of active subject occurrences
        - victim_subj_passive: Count of passive subject occurrences
        - victim_object: Count of object occurrences
        - victim_agency_idx: Agency index in [-1, +1]
        - victim_passive_ratio: Ratio of passive constructions
        - victim_object_ratio: Ratio of object roles
        - victim_role_entropy: Entropy of role distribution [0, 1]
        - victim_agency_class: Classification (alta/media/bassa)
    """
    results = []

    for sentences in spacy_views:
        # Initialize counters
        subj_active = 0
        subj_passive = 0
        obj_count = 0

        # Process each sentence
        for sent in sentences:
            victim_mentions, _ = _detect_entities_in_sentence(sent)

            for mention in victim_mentions:
                role = _classify_role(mention['dep'])

                if role == 'subj_active':
                    subj_active += 1
                elif role == 'subj_passive':
                    subj_passive += 1
                elif role == 'object':
                    obj_count += 1

        # Calculate metrics
        total = subj_active + subj_passive + obj_count

        if total == 0:
            # No victim mentions found
            results.append({
                'victim_subj_active': 0,
                'victim_subj_passive': 0,
                'victim_object': 0,
                'victim_agency_idx': 0.0,
                'victim_passive_ratio': 0.0,
                'victim_object_ratio': 0.0,
                'victim_role_entropy': 0.0,
                'victim_agency_class': 'media',
            })
            continue

        # Ratios
        passive_ratio = subj_passive / total
        object_ratio = obj_count / total

        # Agency index: [-1, +1]
        # +1 = all active subjects
        # -1 = all passive/objects
        agency_idx = (subj_active - subj_passive - obj_count) / total

        # Entropy (normalized to [0, 1])
        probs = np.array([subj_active, subj_passive, obj_count]) / total
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(3)  # Max for 3 categories
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Classification
        if agency_idx > 0.3:
            agency_class = 'alta'
        elif agency_idx < -0.3:
            agency_class = 'bassa'
        else:
            agency_class = 'media'

        results.append({
            'victim_subj_active': subj_active,
            'victim_subj_passive': subj_passive,
            'victim_object': obj_count,
            'victim_agency_idx': agency_idx,
            'victim_passive_ratio': passive_ratio,
            'victim_object_ratio': object_ratio,
            'victim_role_entropy': norm_entropy,
            'victim_agency_class': agency_class,
        })

    return pd.DataFrame(results)


# ==================== NARRATIVE FOCUS ====================

def compute_narrative_focus(spacy_views):
    """
    Compute narrative focus based on weighted mention counts.

    Weights:
    - Subject (active): 1.0
    - Subject (passive): 0.7
    - Object: 0.5

    Args:
        spacy_views: List of spaCy sentence lists per article

    Returns:
        DataFrame with columns:
        - focus_victim_score: Weighted victim score
        - focus_perp_score: Weighted perpetrator score
        - focus_rulebased: Classification (vittima/carnefice/bilanciato)
    """
    results = []

    for sentences in spacy_views:
        victim_score = 0.0
        perp_score = 0.0

        # Process each sentence
        for sent in sentences:
            victim_mentions, perp_mentions = _detect_entities_in_sentence(sent)

            # Score victim mentions
            for mention in victim_mentions:
                role = _classify_role(mention['dep'])

                if role == 'subj_active':
                    victim_score += 1.0
                elif role == 'subj_passive':
                    victim_score += 0.7
                elif role == 'object':
                    victim_score += 0.5

            # Score perpetrator mentions
            for mention in perp_mentions:
                role = _classify_role(mention['dep'])

                if role == 'subj_active':
                    perp_score += 1.0
                elif role == 'subj_passive':
                    perp_score += 0.7
                elif role == 'object':
                    perp_score += 0.5

        # Determine focus
        total_score = victim_score + perp_score

        if total_score == 0:
            focus_label = 'bilanciato'
        else:
            victim_ratio = victim_score / total_score

            if victim_ratio > 0.6:
                focus_label = 'vittima'
            elif victim_ratio < 0.4:
                focus_label = 'carnefice'
            else:
                focus_label = 'bilanciato'

        results.append({
            'focus_victim_score': victim_score,
            'focus_perp_score': perp_score,
            'focus_rulebased': focus_label,
        })

    return pd.DataFrame(results)


# ==================== APPLY TO DATAFRAME ====================

def apply_roles_focus(df, views):
    """
    Apply victim agency and narrative focus detection to dataframe.
    Uses only spaCy views (most reliable for dependency parsing).

    Args:
        df: Input dataframe
        views: Dictionary with 'spacy' views

    Returns:
        DataFrame with added columns:
        - victim_subj_active
        - victim_subj_passive
        - victim_object
        - victim_agency_idx
        - victim_passive_ratio
        - victim_object_ratio
        - victim_role_entropy
        - victim_agency_class
        - focus_victim_score
        - focus_perp_score
        - focus_rulebased
    """
    print("[ROLES & FOCUS] Computing victim agency...")
    agency_df = compute_victim_agency(views["spacy"])

    print("[ROLES & FOCUS] Computing narrative focus...")
    focus_df = compute_narrative_focus(views["spacy"])

    # Merge results
    for col in agency_df.columns:
        df[col] = agency_df[col].values

    for col in focus_df.columns:
        df[col] = focus_df[col].values

    # Print statistics
    print(f"[ROLES & FOCUS] Agency distribution:")
    agency_dist = df['victim_agency_class'].value_counts()
    for label in ['alta', 'media', 'bassa']:
        count = agency_dist.get(label, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"[ROLES & FOCUS] Focus distribution:")
    focus_dist = df['focus_rulebased'].value_counts()
    for label in ['vittima', 'carnefice', 'bilanciato']:
        count = focus_dist.get(label, 0)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("[ROLES & FOCUS] Done.")

    return df