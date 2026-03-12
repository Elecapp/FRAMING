# src/preprocessing/text_preprocess.py
# Text normalisation pipeline producing two representations per article:
#
#   text_linguistic — parser-friendly normalisation for spaCy / Stanza /
#                     regex framing detectors.  Preserves the syntactic and
#                     typographic signals that dependency parsers rely on:
#                     sentence-boundary punctuation (. ! ?), commas (used by
#                     spaCy for clause segmentation), hyphens (compounds),
#                     and apostrophes (clitics / possessives).
#
#   text_ml         — fully-flattened lowercase representation for TF-IDF
#                     classifiers.  All punctuation is removed.
#
# Both functions are pure (no side-effects) and safe to call in parallel.

from __future__ import annotations

import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# Compiled regex patterns  (module-level → compiled once)
# ---------------------------------------------------------------------------

_RE_WS       = re.compile(r"\s+")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL      = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL    = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")

# Age expressions: "26enne", "26-enne", "26 enne"  (case-insensitive)
_RE_AGE = re.compile(r"\b\d+\s*-?\s*enne\b", re.IGNORECASE)

# Italian calendar dates: "31 marzo 2020"  (day month year)
_MONTHS_IT = (
    "gennaio|febbraio|marzo|aprile|maggio|giugno|"
    "luglio|agosto|settembre|ottobre|novembre|dicembre"
)
_RE_DATE_DMY = re.compile(
    rf"\b([0-3]?\d)\s+({_MONTHS_IT})\s+(\d{{4}})\b",
    re.IGNORECASE,
)

# Standalone numeric tokens — removed after placeholders are inserted so that
# bare numbers that carry no framing signal do not inflate the vocabulary.
# Note: digits embedded inside words (e.g. placeholders) are NOT touched
# because \b anchors require a word boundary.
_RE_NUM = re.compile(r"\b\d+\b")

# Punctuation to strip in the linguistic representation.
# Kept  : word chars (\w), whitespace (\s), sentence-boundary markers (. ! ?),
#         comma (,) for clause segmentation, hyphen (-) for compounds,
#         apostrophe (') for clitics, underscore (_) for placeholders.
# Removed: everything else (e.g. brackets, @, #, |, ^, …).
_RE_PUNCT_LINGUISTIC = re.compile(r"[^\w\s.,!?\-']")

# Punctuation to strip in the ML representation: everything except word chars
# and whitespace.  Underscores are kept so _url_ / _age_ / … survive.
_RE_PUNCT_ML = re.compile(r"[^\w\s]")

# Typographic quotes / apostrophes / non-breaking space → ASCII equivalents
_QUOTE_MAP = str.maketrans({
    "\u2018": "'",   # '  left single quotation mark
    "\u2019": "'",   # '  right single quotation mark
    "\u201c": '"',   # "  left double quotation mark
    "\u201d": '"',   # "  right double quotation mark
    "\u0060": "'",   # `  grave accent used as apostrophe
    "\u00a0": " ",   # non-breaking space
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_str(value: object) -> str:
    """
    Safely convert any value to ``str``.

    Handles ``None`` and ``float`` NaN (produced by pandas for empty cells)
    without importing pandas.
    """
    if value is None:
        return ""
    try:
        if value != value:   # NaN is the only value not equal to itself
            return ""
    except TypeError:
        pass
    return str(value)


def _base_normalize(text: str) -> str:
    """
    Apply HTML unescaping, Unicode normalisation, and quote mapping.

    Shared by both downstream representations so all patterns downstream
    see a clean, canonical character set.
    """
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _RE_HTML_TAG.sub(" ", text)
    text = text.translate(_QUOTE_MAP)
    return text


def _clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading / trailing spaces."""
    return _RE_WS.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_text_linguistic(title: object, body: object) -> str:
    """
    Build the parser-friendly normalised representation.

    Designed for spaCy / Stanza dependency parsing and regex-based framing
    detectors.  Preserves the punctuation signals that parsers rely on for
    sentence segmentation and syntactic analysis, while removing noise that
    is irrelevant to framing detection (HTML, URLs, bare numbers, etc.).

    Preservation rationale
    ~~~~~~~~~~~~~~~~~~~~~~
    - Sentence-boundary markers (. ! ?) — required for sentence segmentation.
    - Commas (,) — used by spaCy to split clauses and appositions; removing
      them would break constructions such as "Mario, accusato di...".
    - Hyphens (-) — needed for compound adjectives and hyphenated expressions
      common in Italian journalism.
    - Apostrophes (') — mark clitics and elisions (e.g. "l'uomo", "dell'ex");
      removing them would corrupt tokenisation.

    Steps
    -----
    1. Concatenate title and body with a blank-line separator.
    2. HTML unescaping, Unicode normalisation, quote normalisation.
    3. Replace structured patterns with domain placeholders:
       - URLs        → ``_url_``
       - e-mail      → ``_email_``
       - Age tokens  → ``_age_``
       - Dates DMY   → ``_date_``
       - Bare numbers → removed
    4. Strip punctuation except sentence markers, commas, hyphens,
       apostrophes, and underscores.
    5. Collapse whitespace.

    Parameters
    ----------
    title : object
        Article title (str, None, or float NaN).
    body : object
        Article body — the ``cleaned_text`` column
        (str, None, or float NaN).

    Returns
    -------
    str
        Normalised text suitable for linguistic parsing.
    """
    title = _to_str(title)
    body  = _to_str(body)

    if title and body:
        text = f"{title}\n\n{body}"
    else:
        text = title or body

    if not text:
        return ""

    text = _base_normalize(text)

    # Domain-specific placeholder substitutions (order matters:
    # email before URL to avoid partial URL matches swallowing addresses)
    text = _RE_EMAIL.sub(" _email_ ", text)
    text = _RE_URL.sub(" _url_ ", text)
    text = _RE_AGE.sub(" _age_ ", text)
    text = _RE_DATE_DMY.sub(" _date_ ", text)
    text = _RE_NUM.sub(" ", text)

    # Strip non-linguistic punctuation
    text = _RE_PUNCT_LINGUISTIC.sub(" ", text)

    return _clean_whitespace(text)


def build_text_ml(text_linguistic: object) -> str:
    """
    Build the fully-flattened ML representation from ``text_linguistic``.

    Used as input to TF-IDF vectorisers.  All punctuation (including
    sentence markers and commas) is removed and the text is lowercased.
    Placeholders inserted by :func:`build_text_linguistic` survive because
    they consist only of word characters and underscores.

    Parameters
    ----------
    text_linguistic : object
        Output of :func:`build_text_linguistic` (str, None, or float NaN).

    Returns
    -------
    str
        Lowercase text without any punctuation.
    """
    text = _to_str(text_linguistic)
    if not text:
        return ""

    text = text.lower()
    text = _RE_PUNCT_ML.sub(" ", text)

    return _clean_whitespace(text)