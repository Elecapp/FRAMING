# src/framing_rule_based/lexicons.py
"""
Lexicons for rule-based framing detection.
All lexical resources used across victim-blaming, justification, and tone modules.
"""

# ==================== VICTIM-BLAMING ====================

VICTIM_TERMS = {
    "vittima", "donna", "ragazza", "moglie",
    "compagna", "fidanzata", "persona", "persona offesa",
}

BLAME_PHRASES = {
    "se l'è cercata",
    "se l' è cercata",
    "se l è cercata",
    "colpa sua",
    "gliel'ha cercata",
    "se l'ha cercata",
    "se la è cercata",
    "è colpa sua",
    "se l'avesse evitato",
}

BLAME_VERBS = {
    "provocare", "provocava", "provocato",
    "cercare", "cercava", "cercato",
    "tradire", "tradiva",
    "insultare", "stuzzicare", "sedurre",
}

BLAME_NGRAMS_2 = {
    "foto provocanti",
    "intimo provocante",
    "lo tradiva",
    "ragazza libera",
    "era ubriaca",
}

BLAME_NGRAMS_3 = {
    "scambiava foto sexy",
    "non curava la casa",
    "lo aveva lasciato",
    "lo voleva lasciare",
    "non aveva denunciato",
}

BLAME_NGRAMS_4 = {
    "lei voleva la separazione",
    "non sopportava il tradimento",
    "lite degenerata in tragedia",
}


# ==================== JUSTIFICATION ====================

JUST_EMOTION = {
    "gelosia", "geloso", "gelosa",
    "invidia", "rabbia", "impeto",
}

JUST_MENTAL = {
    "raptus", "follia", "pazzia",
    "depressione", "depresso", "depressa",
    "stress", "esaurimento",
}

JUST_CIRCUMSTANCE = {
    "problemi", "economici", "familiari",
    "lavoro", "licenziamento", "crisi", "debiti",
}

JUST_PHRASES = {
    "momento di follia",
    "perdita di controllo",
    "colpo di testa",
    "non si è reso conto",
    "non si e reso conto",
}

JUST_ALL_TERMS = JUST_EMOTION | JUST_MENTAL | JUST_CIRCUMSTANCE

PERP_TERMS = {
    "marito", "ex", "ex marito",
    "fidanzato", "ex fidanzato",
    "compagno", "ex compagno",
    "uomo", "assassino", "omicida",
    "killer", "aggressore", "imputato", "indagato",
}


# ==================== TONE ====================

TONE_EMOTIVE_TERMS = {
    "cordoglio", "commozione", "sgomento", "sconcerto",
    "dolore", "profondo dolore", "lutto", "grave lutto",
    "tragedia", "tragico", "tragica",
    "scioccati", "scioccata", "sconvolti", "sconvolta", "sconvolgente",
    "vicinanza",
    "si stringe attorno", "si stringono attorno",
    "abbraccio", "abbraccia la famiglia",
    "comunità in lutto", "comunità sotto shock", "comunità sconvolta",
    "shock in paese", "shock in città",
}

TONE_EMOTIVE_PHRASES = {
    "il paese è sotto choc",
    "il paese è sotto shock",
    "la comunità è sotto choc",
    "la comunità è sotto shock",
    "profondo cordoglio",
    "cordoglio della comunità",
    "cordoglio delle istituzioni",
    "stretti attorno alla famiglia",
    "si stringe attorno alla famiglia",
}

TONE_EMOTIVE_VERBS = {
    "piangere", "commuovere", "stringersi", "abbracciare", "soffrire",
}

TONE_EMOTIVE_CONTEXTS = {
    "famiglia", "parenti", "comunità", "paese", "città",
}

TONE_SENSATIONAL_TERMS = {
    "orrore", "orribile", "terribile", "agghiacciante",
    "choc", "shock", "massacro", "macello", "mattanza",
    "bagno di sangue", "inferno", "incubo", "incubo senza fine",
    "dramma", "drammatico", "drammatica",
    "horror", "giallo", "giallo irrisolto",
    "delitto choc", "omicidio choc", "femminicidio choc",
}

TONE_SENSATIONAL_PHRASES = {
    "orrore senza fine",
    "notte di orrore",
    "scena da film horror",
    "delitto da brividi",
    "omicidio da brividi",
    "un vero e proprio massacro",
    "omicidio shock",
    "femminicidio shock",
    "brividi lungo la schiena",
}

TONE_SENSATIONAL_TYPO = {"!!!", "??", "!?", "?!"}


# ==================== SHARED MODIFIERS ====================

CAUSALS = {
    "perché", "perche",
    "poiché", "poiche", "poichè",
    "a causa di", "per via di", "dovuto a",
}

HEDGES = {
    "pare", "sembra", "forse",
    "avrebbe", "avrebbero",
    "secondo", "stando a", "a detta di",
}


# ==================== ROLES & FOCUS ====================

VICTIM_UNIGRAMS = {
    "vittima", "donna", "ragazza", "moglie",
    "compagna", "fidanzata", "madre", "figlia",
}

VICTIM_MULTI = ["ex moglie", "ex compagna", "ex fidanzata"]

PERP_UNIGRAMS = {
    "assassino", "omicida", "killer", "uomo",
    "marito", "ex", "fidanzato", "compagno",
    "indagato", "arrestato",
}

PERP_MULTI = ["ex marito", "ex compagno", "ex fidanzato"]