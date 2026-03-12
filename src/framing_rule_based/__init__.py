# src/framing_rule_based/__init__.py
"""
Rule-based framing detection module.

Provides lexicon-based and syntax-based analysis for:
- Victim-blaming
- Perpetrator justification
- Tone (emotive vs sensationalistic)
- Victim agency
- Narrative focus
"""

from .lexicons import (
    VICTIM_TERMS,
    BLAME_PHRASES,
    BLAME_VERBS,
    JUST_EMOTION,
    JUST_MENTAL,
    JUST_CIRCUMSTANCE,
    PERP_TERMS,
    TONE_EMOTIVE_TERMS,
    TONE_SENSATIONAL_TERMS,
)

from .parsing import (
    build_rule_based_views,
    select_regex_sentences,
    select_spacy_sentences,
    select_stanza_sentences,
    majority_vote_binary,
    consensus_multiclass,
)

from .scoring import (
    sentence_score_victim_blaming,
    sentence_score_justification,
    sentence_score_tone_emotive,
    sentence_score_tone_sensational,
)

from .victim_blaming import (
    predict_victim_blaming_regex,
    predict_victim_blaming_spacy,
    predict_victim_blaming_stanza,
    apply_victim_blaming,
)

from .justification import (
    predict_justification_regex,
    predict_justification_spacy,
    predict_justification_stanza,
    apply_justification,
)

from .tone import (
    predict_tone_regex,
    predict_tone_spacy,
    predict_tone_stanza,
    apply_tone,
)

from .roles_focus import (
    compute_victim_agency,
    compute_narrative_focus,
    apply_roles_focus,
)

__all__ = [
    # Lexicons
    "VICTIM_TERMS",
    "BLAME_PHRASES",
    "BLAME_VERBS",
    "JUST_EMOTION",
    "JUST_MENTAL",
    "JUST_CIRCUMSTANCE",
    "PERP_TERMS",
    "TONE_EMOTIVE_TERMS",
    "TONE_SENSATIONAL_TERMS",

    # Parsing
    "build_rule_based_views",
    "select_regex_sentences",
    "select_spacy_sentences",
    "select_stanza_sentences",
    "majority_vote_binary",
    "consensus_multiclass",

    # Scoring
    "sentence_score_victim_blaming",
    "sentence_score_justification",
    "sentence_score_tone_emotive",
    "sentence_score_tone_sensational",

    # Victim-blaming
    "predict_victim_blaming_regex",
    "predict_victim_blaming_spacy",
    "predict_victim_blaming_stanza",
    "apply_victim_blaming",

    # Justification
    "predict_justification_regex",
    "predict_justification_spacy",
    "predict_justification_stanza",
    "apply_justification",

    # Tone
    "predict_tone_regex",
    "predict_tone_spacy",
    "predict_tone_stanza",
    "apply_tone",

    # Roles & Focus
    "compute_victim_agency",
    "compute_narrative_focus",
    "apply_roles_focus",
]