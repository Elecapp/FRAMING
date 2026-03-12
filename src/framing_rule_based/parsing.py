# src/framing_rule_based/parsing.py
"""
Shared parsing utilities for rule-based framing detection.
Builds linguistic views (regex, spaCy, Stanza) from text and provides
sentence selection utilities.
"""

import re
from tqdm import tqdm
import spacy

# ==================== REGEX SENTENCE SPLITTER ====================

SENT_SPLIT_RE = re.compile(r"[.!?]+")


def split_sentences_regex(text):
    """
    Split text into sentences using simple regex pattern.

    Args:
        text: Input text string

    Returns:
        List of sentence strings
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


# ==================== STANZA INITIALIZATION ====================

try:
    import stanza

    NLP_STANZA = stanza.Pipeline(
        "it",
        processors="tokenize,mwt,pos,lemma,depparse",
        use_gpu=True,
        verbose=False,
        download_method=None,
    )
except ImportError:
    raise RuntimeError(
        "[FRAMING][STANZA] Stanza not installed. "
        "Install with: pip install stanza"
    )
except Exception as e:
    raise RuntimeError(
        "[FRAMING][STANZA] Could not initialize Stanza pipeline. "
        "Make sure the Italian model is downloaded with: "
        "python -c 'import stanza; stanza.download(\"it\")'"
    ) from e

# ==================== SPACY INITIALIZATION ====================

try:
    # it_core_news_lg includes the dependency parser needed for roles_focus.py.
    # it_core_news_sm only has a tagger/morphologizer — dep_ labels are empty,
    # which causes victim agency and narrative focus to return all-zero.
    NLP_SPACY = spacy.load("it_core_news_lg")
except OSError:
    try:
        NLP_SPACY = spacy.load("it_core_news_sm")
        print(
            "[FRAMING][SPACY] WARNING: it_core_news_lg not found, falling back to sm. "
            "Dependency parsing (roles_focus) will not work correctly. "
            "Install with: python -m spacy download it_core_news_lg"
        )
    except OSError:
        NLP_SPACY = spacy.blank("it")
        print("[FRAMING][SPACY] Using blank Italian model (no pretrained weights)")

# Do NOT add sentencizer when the parser is present — the parser handles
# sentence segmentation and adding sentencizer would conflict with it.
if (
    "parser" not in NLP_SPACY.pipe_names
    and "sentencizer" not in NLP_SPACY.pipe_names
    and "senter" not in NLP_SPACY.pipe_names
):
    NLP_SPACY.add_pipe("sentencizer")

# Increase max length for long articles
NLP_SPACY.max_length = 2_000_000


# ==================== BUILD VIEWS ====================

def build_rule_based_views(texts, batch_size=128, show_progress=True):
    """
    Parse texts with regex, spaCy, and Stanza to create reusable views.

    Args:
        texts: pandas Series or list of strings
        batch_size: Batch size for spaCy processing
        show_progress: Show progress bars

    Returns:
        Dictionary with keys 'regex', 'spacy', 'stanza' containing
        lists of parsed sentence objects
    """
    # Convert to list of strings
    if hasattr(texts, 'fillna'):
        texts = texts.fillna("").astype(str).tolist()
    else:
        texts = [str(t) if t else "" for t in texts]

    # ===== REGEX =====
    regex_views = []
    iterator = tqdm(texts, desc="Parsing (regex)", disable=not show_progress)
    for text in iterator:
        regex_views.append(split_sentences_regex(text))

    # ===== SPACY =====
    spacy_views = []
    iterator = NLP_SPACY.pipe(texts, batch_size=batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=len(texts), desc="Parsing (spaCy)")

    for doc in iterator:
        spacy_views.append(list(doc.sents))

    # ===== STANZA =====
    stanza_views = []
    iterator = tqdm(texts, desc="Parsing (Stanza)", disable=not show_progress)
    for text in iterator:
        try:
            doc = NLP_STANZA(text)
            stanza_views.append(doc.sentences)
        except Exception as e:
            # Fallback to empty list on parsing errors
            print(f"[STANZA] Parse error: {e}")
            stanza_views.append([])

    return {
        "regex": regex_views,
        "spacy": spacy_views,
        "stanza": stanza_views,
    }


# ==================== SENTENCE SELECTION ====================

def _trim_sentences(sentences, char_limit, max_sentences):
    """
    Select sentences respecting character limit and max count.

    Args:
        sentences: List of sentence objects (str, spaCy Span, or Stanza Sentence)
        char_limit: Maximum cumulative characters
        max_sentences: Maximum number of sentences

    Returns:
        List of selected sentences (same type as input)
    """
    if not sentences:
        return []

    kept = []
    total_chars = 0

    for sent in sentences:
        # Extract text based on object type
        if hasattr(sent, "text"):
            text = sent.text
        else:
            text = str(sent)

        text_len = len(text)

        # Check character limit
        if total_chars + text_len > char_limit:
            break

        kept.append(sent)
        total_chars += text_len

        # Check sentence limit
        if max_sentences and len(kept) >= max_sentences:
            break

    # Always keep at least first sentence if available
    if not kept and sentences:
        kept = [sentences[0]]

    return kept


def select_regex_sentences(sentences, char_limit, max_sentences):
    """Select sentences from regex view."""
    return _trim_sentences(sentences, char_limit, max_sentences)


def select_spacy_sentences(sentences, char_limit, max_sentences):
    """Select sentences from spaCy view."""
    return _trim_sentences(sentences, char_limit, max_sentences)


def select_stanza_sentences(sentences, char_limit, max_sentences):
    """Select sentences from Stanza view."""
    return _trim_sentences(sentences, char_limit, max_sentences)


# ==================== CONSENSUS UTILITIES ====================

def majority_vote_binary(a, b, c):
    """
    Binary majority vote (2 out of 3).

    Args:
        a, b, c: Binary predictions (0 or 1)

    Returns:
        1 if at least 2 predictions are 1, else 0
    """
    return int(int(a) + int(b) + int(c) >= 2)


def consensus_multiclass(labels):
    """
    Multiclass consensus with tie-breaking.

    Args:
        labels: List of string labels

    Returns:
        Most common label (with priority order for ties)
    """
    from collections import Counter

    # Filter valid labels
    labels = [l for l in labels if isinstance(l, str) and l]
    if not labels:
        return "neutro"

    counts = Counter(labels)
    most_common = counts.most_common()

    # If clear winner, return it
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]

    # Tie-breaking: get all labels with max count
    max_cnt = most_common[0][1]
    candidates = [lab for lab, cnt in most_common if cnt == max_cnt]

    # Priority order for tone
    for preferred in ("emotivo", "sensazionalistico", "neutro"):
        if preferred in candidates:
            return preferred

    # Fallback to first candidate
    return candidates[0]