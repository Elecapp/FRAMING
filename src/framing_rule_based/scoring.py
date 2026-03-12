# src/framing_rule_based/scoring.py
# Sentence-level scoring functions for rule-based framing detection.
#
# Each function takes a sentence object (str, spaCy Span, or Stanza Sentence)
# and returns a float score for a specific framing dimension.
# Higher scores indicate stronger presence of the framing signal.
#
# Scores are NOT probabilities — they are used by the predict_* functions
# to compute article-level ratios, which are then thresholded.

from __future__ import annotations

import re

import spacy.tokens
import stanza.models.common.doc as stanza_doc

from .lexicons import (
    BLAME_NGRAMS_2,
    BLAME_NGRAMS_3,
    BLAME_NGRAMS_4,
    BLAME_PHRASES,
    BLAME_VERBS,
    CAUSALS,
    HEDGES,
    JUST_ALL_TERMS,
    JUST_PHRASES,
    PERP_TERMS,
    TONE_EMOTIVE_CONTEXTS,
    TONE_EMOTIVE_PHRASES,
    TONE_EMOTIVE_TERMS,
    TONE_EMOTIVE_VERBS,
    TONE_SENSATIONAL_PHRASES,
    TONE_SENSATIONAL_TERMS,
    TONE_SENSATIONAL_TYPO,
    VICTIM_TERMS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_text(sent: object) -> str:
    """
    Extract lowercase text from a sentence object.

    Handles str, spaCy Span (``sent.text``), and Stanza Sentence
    (``sent.text``).  All three expose a ``.text`` attribute, so
    ``hasattr(sent, 'text')`` is a reliable discriminator.
    """
    if hasattr(sent, "text"):
        return sent.text.lower()
    return str(sent).lower()


def _get_text_original(sent: object) -> str:
    """Return original-case text (needed for ALL-CAPS detection)."""
    if hasattr(sent, "text"):
        return sent.text
    return str(sent)


def _has_passive(sent: object) -> bool:
    """
    Check whether a sentence contains passive voice.

    Uses morphological features when available (spaCy / Stanza).
    Falls back to False for plain-string sentences.

    spaCy  : checks ``token.morph`` for ``Voice=Pass`` or ``dep_ == 'nsubj:pass'``.
    Stanza : checks ``word.feats`` for ``Voice=Pass``.
    """
    # spaCy Span
    if isinstance(sent, spacy.tokens.Span):
        for token in sent:
            if token.dep_ in {"nsubj:pass", "auxpass"}:
                return True
            if "Pass" in token.morph.get("Voice", []):
                return True
        return False

    # Stanza Sentence
    if isinstance(sent, stanza_doc.Sentence):
        for word in sent.words:
            if word.feats and "Voice=Pass" in word.feats:
                return True
        return False

    return False


def _count_ngrams(text: str, ngrams_set: set, n: int) -> int:
    """Count how many n-grams from ``ngrams_set`` appear in ``text``."""
    words = text.split()
    count = 0
    for i in range(len(words) - n + 1):
        if " ".join(words[i : i + n]) in ngrams_set:
            count += 1
    return count


def _has_negation_window(text: str, term: str, window: int = 5) -> bool:
    """
    Return True if ``term`` appears within ``window`` tokens of a negation word.

    Used to down-weight justification terms that are explicitly denied
    (e.g. "non è gelosia").
    """
    _NEGATIONS = {"non", "nessun", "nessuna", "mai", "niente", "nulla"}
    words = text.split()
    for i, word in enumerate(words):
        if term in word:
            start = max(0, i - window)
            end   = min(len(words), i + window + 1)
            if set(words[start:end]) & _NEGATIONS:
                return True
    return False


# ---------------------------------------------------------------------------
# Victim-blaming scoring
# ---------------------------------------------------------------------------

def sentence_score_victim_blaming(sent: object) -> float:
    """
    Score a sentence for victim-blaming content.

    Scoring weights
    ---------------
    Blame phrases (exact)           +1.0 each
    4-grams                         +1.0 each
    3-grams                         +0.9 each
    2-grams                         +0.6 each
    Blame verb AND victim term      +0.8
    Causal connective               +0.2
    Hedge marker                    +0.1
    Passive voice                   +0.1

    Parameters
    ----------
    sent : str | spaCy Span | Stanza Sentence

    Returns
    -------
    float
        Non-negative score; 0.0 means no signal detected.
    """
    text  = _get_text(sent)
    score = 0.0

    # Exact blame phrases (highest signal)
    for phrase in BLAME_PHRASES:
        if phrase in text:
            score += 1.0

    # N-gram matches (decreasing weight with length)
    score += _count_ngrams(text, BLAME_NGRAMS_4, 4) * 1.0
    score += _count_ngrams(text, BLAME_NGRAMS_3, 3) * 0.9
    score += _count_ngrams(text, BLAME_NGRAMS_2, 2) * 0.6

    # Blame verb co-occurring with victim term
    if any(v in text for v in BLAME_VERBS) and any(t in text for t in VICTIM_TERMS):
        score += 0.8

    # Discourse modifiers
    if any(c in text for c in CAUSALS):
        score += 0.2
    if any(h in text for h in HEDGES):
        score += 0.1
    if _has_passive(sent):
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# Perpetrator-justification scoring
# ---------------------------------------------------------------------------

def sentence_score_justification(sent: object) -> float:
    """
    Score a sentence for perpetrator justification.

    Scoring weights
    ---------------
    Justification phrase (exact)    +1.0 each
    Justification term              +0.7 (−0.6 if negated within window 5)
    Causal connective               +0.3
    Hedge marker                    +0.2
    Perpetrator term                +0.2
    Passive voice                   +0.1

    Note: the score can be negative when justification terms are strongly
    negated.  Downstream thresholding handles this correctly because the
    default ``sent_threshold`` is positive.

    Parameters
    ----------
    sent : str | spaCy Span | Stanza Sentence

    Returns
    -------
    float
    """
    text  = _get_text(sent)
    score = 0.0

    # Exact justification phrases
    for phrase in JUST_PHRASES:
        if phrase in text:
            score += 1.0

    # Justification terms with negation check
    for term in JUST_ALL_TERMS:
        if term in text:
            if _has_negation_window(text, term, window=5):
                score -= 0.6
            else:
                score += 0.7

    # Discourse modifiers
    if any(c in text for c in CAUSALS):
        score += 0.3
    if any(h in text for h in HEDGES):
        score += 0.2
    if any(p in text for p in PERP_TERMS):
        score += 0.2
    if _has_passive(sent):
        score += 0.1

    return score


# ---------------------------------------------------------------------------
# Tone scoring — emotive
# ---------------------------------------------------------------------------

def sentence_score_tone_emotive(sent: object) -> float:
    """
    Score a sentence for emotive tone.

    Scoring weights
    ---------------
    Emotive phrase (exact)                          +1.0 each
    Emotive term                                    +0.7 each
    Emotive verb AND community/family context       +0.5
    Emotive verb alone                              +0.3

    Parameters
    ----------
    sent : str | spaCy Span | Stanza Sentence

    Returns
    -------
    float  (≥ 0.0)
    """
    text  = _get_text(sent)
    score = 0.0

    for phrase in TONE_EMOTIVE_PHRASES:
        if phrase in text:
            score += 1.0

    for term in TONE_EMOTIVE_TERMS:
        if term in text:
            score += 0.7

    has_verb    = any(v in text for v in TONE_EMOTIVE_VERBS)
    has_context = any(c in text for c in TONE_EMOTIVE_CONTEXTS)

    if has_verb and has_context:
        score += 0.5
    elif has_verb:
        score += 0.3

    return score


# ---------------------------------------------------------------------------
# Tone scoring — sensationalistic
# ---------------------------------------------------------------------------

def sentence_score_tone_sensational(sent: object) -> float:
    """
    Score a sentence for sensationalistic tone.

    Scoring weights
    ---------------
    Sensational phrase (exact)      +1.0 each
    Sensational term                +0.7 each
    Punctuation pattern (!!!, ?!)   +0.3
    ALL-CAPS word (≥ 4 chars)       +0.3
    Superlative suffix (-issimo)    +0.2

    Parameters
    ----------
    sent : str | spaCy Span | Stanza Sentence

    Returns
    -------
    float  (≥ 0.0)
    """
    text_orig = _get_text_original(sent)
    text      = text_orig.lower()
    score     = 0.0

    for phrase in TONE_SENSATIONAL_PHRASES:
        if phrase in text:
            score += 1.0

    for term in TONE_SENSATIONAL_TERMS:
        if term in text:
            score += 0.7

    # Punctuation excess
    for pattern in TONE_SENSATIONAL_TYPO:
        if pattern in text_orig:
            score += 0.3

    # ALL-CAPS words (≥ 4 consecutive capital letters)
    if re.search(r'\b[A-Z]{4,}\b', text_orig):
        score += 0.3

    # Italian superlative suffix
    if re.search(r'\w+issim[oiae]\b', text):
        score += 0.2

    return score