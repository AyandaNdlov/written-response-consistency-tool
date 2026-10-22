"""Feature extraction for written responses.

Everything here is plain Python and regex (no spaCy or torch) so the app stays
light enough for Streamlit Community Cloud. All functions work on strings in
memory; nothing is written to disk.
"""

from __future__ import annotations

import math
import re
import statistics

import textstat

# --------------------------------------------------------------------------
# Word lists
# --------------------------------------------------------------------------

# ~60 high-frequency English function words. Their relative frequencies are a
# classic stylometric "fingerprint" because people use them largely unconsciously.
FUNCTION_WORDS = [
    "the", "a", "an", "and", "but", "or", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "about", "as", "into", "that", "this", "these",
    "those", "which", "who", "what", "when", "where", "it", "its", "is", "was",
    "be", "been", "are", "were", "have", "has", "had", "not", "no", "so", "if",
    "then", "than", "there", "would", "could", "should", "will", "can", "may",
    "also", "very", "just", "all", "some", "more", "such", "because", "while",
]

# Connectives and stock phrases that are common in formal, templated or
# generated prose. A high rate is not evidence of anything on its own.
DISCOURSE_MARKERS = [
    "furthermore", "moreover", "additionally", "in addition", "in conclusion",
    "to conclude", "in summary", "it is important to note", "it's important to note",
    "it is worth noting", "overall", "delve", "delves", "delving",
    "crucial", "crucially", "notably", "in today's", "ultimately", "consequently",
    "nevertheless", "firstly", "secondly", "thirdly", "lastly", "pivotal",
    "foster", "fostering", "leverage", "leveraging", "navigate", "navigating",
    "seamless", "seamlessly", "holistic", "robust", "tapestry", "landscape",
    "by doing so", "plays a vital role",
]
# Phrases must not overlap (e.g. "it is crucial" and "crucial"), or they double-count.

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}

CONTRACTION_RE = re.compile(r"(n't|'re|'ve|'ll|'d|'m)$")
S_CONTRACTIONS = {
    "it's", "that's", "there's", "what's", "he's", "she's", "let's", "here's",
    "who's", "where's", "how's", "everyone's", "nobody's",
}

# --- British vs American spelling ---------------------------------------

_OUR_ROOTS = [
    "behaviour", "colour", "favour", "honour", "labour", "neighbour", "endeavour",
    "humour", "rumour", "flavour", "harbour", "vigour", "savour", "valour",
]
_OUR_SUFFIXES = ["", "s", "ed", "ing", "al", "ally", "ist", "ists", "ite", "ites",
                 "able", "ably", "ful", "less", "hood", "hoods", "er", "ers"]

_EXPLICIT_PAIRS = {
    # -re / -er
    "centre": "center", "centres": "centers", "centred": "centered",
    "theatre": "theater", "theatres": "theaters", "fibre": "fiber",
    "litre": "liter", "litres": "liters", "metre": "meter", "metres": "meters",
    # programme
    "programme": "program", "programmes": "programs",
    # -yse / -yze ("analyses" is left out: it is also the plural of "analysis")
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "paralyse": "paralyze", "paralysed": "paralyzed", "catalyse": "catalyze",
    # -ce / -se
    "defence": "defense", "offence": "offense", "pretence": "pretense",
    # doubled l
    "travelled": "traveled", "travelling": "traveling", "traveller": "traveler",
    "travellers": "travelers", "cancelled": "canceled", "cancelling": "canceling",
    "modelling": "modeling", "modelled": "modeled", "labelled": "labeled",
    "labelling": "labeling", "signalled": "signaled", "counselling": "counseling",
    "counsellor": "counselor", "fuelled": "fueled", "levelled": "leveled",
    "enrolment": "enrollment", "fulfilment": "fulfillment", "skilful": "skillful",
    # misc
    "grey": "gray", "cheque": "check", "jewellery": "jewelry", "catalogue": "catalog",
    "manoeuvre": "maneuver", "ageing": "aging", "aluminium": "aluminum",
}

BRITISH_TO_AMERICAN: dict[str, str] = dict(_EXPLICIT_PAIRS)
for _root in _OUR_ROOTS:
    _amer_root = _root[: _root.rfind("our")] + "or"
    for _suf in _OUR_SUFFIXES:
        BRITISH_TO_AMERICAN[_root + _suf] = _amer_root + _suf
AMERICAN_TO_BRITISH = {v: k for k, v in BRITISH_TO_AMERICAN.items()}

# -ise / -ize verbs built on common stems (e.g. organise, prioritise, realise).
_ISE_STEMS = [
    "organ", "recogn", "priorit", "real", "util", "minim", "maxim", "optim",
    "summar", "categor", "emphas", "special", "standard", "critic", "apolog",
    "character", "familiar", "general", "visual", "mobil", "capital", "stabil",
    "harmon", "modern", "normal", "personal", "custom", "memor", "symbol",
    "energ", "author", "final", "central", "global", "local", "social",
    "rational", "conceptual", "contextual", "internal", "external", "legal",
    "human", "sympath", "empath", "theor", "polar", "jeopard", "patron",
    "synchron", "digit", "monet", "incentiv", "revital", "sanit", "dramat",
    "scrutin", "commercial", "industrial", "individual", "marginal", "neutral",
    "trivial", "material", "vocal", "formal", "civil", "util", "subsid",
]
ISE_IZE_RE = re.compile(
    r"\b(" + "|".join(sorted(set(_ISE_STEMS), key=len, reverse=True)) + r")"
    r"(is|iz)(e|es|ed|ing|ation|ations|er|ers)\b"
)

ABBREVIATIONS = ["e.g.", "i.e.", "etc.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.",
                 "vs.", "St.", "No.", "approx."]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Keys of the scalar features that are compared between texts.
SCALAR_FEATURES = [
    "sentence_len_mean", "sentence_len_sd", "avg_word_len", "mtld",
    "long_word_pct", "commas_p100", "semicolons_colons_p100", "dashes_p100",
    "excl_quest_p100", "parentheses_p100", "flesch_reading_ease", "fk_grade",
    "discourse_markers_p100", "british_share", "contractions_p100",
    "first_person_p100",
]


# --------------------------------------------------------------------------
# Text splitting
# --------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Standardise line endings, apostrophes and quotes."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub("[’‘ʼ`]", "'", text)
    text = re.sub("[“”]", '"', text)
    return text


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines.

    If the text has no blank lines but has single line breaks that look like
    paragraph ends (most lines finish with sentence punctuation), split on those
    instead. Hard-wrapped text (lines ending mid-sentence) is kept together.
    """
    text = normalise(text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1 and "\n" in paras[0]:
        lines = [ln.strip() for ln in paras[0].split("\n") if ln.strip()]
        ends = sum(ln[-1] in ".!?:\"')" for ln in lines)
        if ends / len(lines) >= 0.6:
            paras = lines
        else:
            paras = [" ".join(lines)]
    return [re.sub(r"\s*\n\s*", " ", p) for p in paras]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, paragraph by paragraph.

    Abbreviations such as "e.g." are protected first so their full stops are
    not mistaken for sentence ends.
    """
    sentences: list[str] = []
    for para in split_paragraphs(text):
        protected = para
        for abbr in ABBREVIATIONS:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
        # Split after . ! or ?, allowing closing quotes or brackets before the space.
        for part in re.split(r"(?<=[.!?])[\"')\]]*\s+", protected):
            part = part.replace("<DOT>", ".").strip()
            if WORD_RE.search(part):
                sentences.append(part)
    return sentences


def tokenize(text: str) -> list[str]:
    """Return the words in a text, keeping apostrophes inside words (don't, it's)."""
    return WORD_RE.findall(normalise(text))


# --------------------------------------------------------------------------
# Individual measures
# --------------------------------------------------------------------------

def _mtld_pass(tokens: list[str], threshold: float = 0.72) -> float:
    """One directional MTLD pass.

    Walk through the words, tracking the type-token ratio. Each time it drops to
    the threshold, count one "factor" and start again. The leftover stretch at the
    end counts as a partial factor. MTLD is the number of words per factor.
    """
    factors = 0.0
    types: set[str] = set()
    count = 0
    for tok in tokens:
        count += 1
        types.add(tok)
        if len(types) / count <= threshold:
            factors += 1
            types = set()
            count = 0
    if count > 0:
        ttr = len(types) / count
        factors += (1 - ttr) / (1 - threshold)
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010).

    Mean of a forward and a backward pass. Unlike a raw type-token ratio it is
    fairly stable across text lengths, which matters when comparing a short
    supervised sample with a longer remote response.
    """
    tokens = [t.lower() for t in tokens]
    if not tokens:
        return 0.0
    return (_mtld_pass(tokens, threshold) + _mtld_pass(tokens[::-1], threshold)) / 2


def spelling_counts(text: str) -> tuple[int, int]:
    """Return (british, american) spelling-variant counts."""
    lower = normalise(text).lower()
    british = american = 0
    for tok in WORD_RE.findall(lower):
        if tok in BRITISH_TO_AMERICAN:
            british += 1
        elif tok in AMERICAN_TO_BRITISH:
            american += 1
    for match in ISE_IZE_RE.finditer(lower):
        if match.group(2) == "is":
            british += 1
        else:
            american += 1
    return british, american


def discourse_marker_counts(text: str) -> dict[str, int]:
    """Count each stock connective or phrase that appears, as whole words only."""
    lower = normalise(text).lower()
    found = {}
    for marker in DISCOURSE_MARKERS:
        n = len(re.findall(r"\b" + re.escape(marker) + r"\b", lower))
        if n:
            found[marker] = n
    return found


def function_word_profile(tokens: list[str]) -> dict[str, float]:
    """Relative frequency of each function word (its share of all words)."""
    lower = [t.lower() for t in tokens]
    n = len(lower) or 1
    return {w: lower.count(w) / n for w in FUNCTION_WORDS}


def function_word_distance(profile_a: dict[str, float], profile_b: dict[str, float]) -> float:
    """Cosine distance (0 = identical profile, 1 = nothing in common)."""
    a = [profile_a.get(w, 0.0) for w in FUNCTION_WORDS]
    b = [profile_b.get(w, 0.0) for w in FUNCTION_WORDS]
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 1.0
    return max(0.0, 1.0 - dot / norm)


def is_contraction(token: str) -> bool:
    """True for forms such as don't, I'm and it's, but not possessives like John's."""
    t = token.lower()
    return bool(CONTRACTION_RE.search(t)) or t in S_CONTRACTIONS


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def extract_features(text: str) -> dict:
    """Extract every writing feature used by the tool from one text."""
    text = normalise(text)
    tokens = tokenize(text)
    n = len(tokens)

    # Most features are rates per 100 words, so texts of different lengths compare fairly.
    def per100(count: float) -> float:
        return 100.0 * count / n if n else 0.0

    # Sentence lengths (in words) and word lengths (in letters, ignoring apostrophes).
    sent_lens = [len(tokenize(s)) for s in split_sentences(text)]
    sent_lens = [s for s in sent_lens if s > 0]
    letters = [len(t.replace("'", "")) for t in tokens]

    # Dashes used as punctuation: em dashes, en dashes, and hyphens with a space on each
    # side (" - " or " -- "). Hyphens inside words such as "follow-up" are not counted.
    dashes = (text.count("\u2014") + text.count("\u2013")
              + len(re.findall(r"(?<=\s)-{1,2}(?=\s)", text)))
    markers = discourse_marker_counts(text)
    british, american = spelling_counts(text)
    # "I'm" and "we've" count as first person, so look at the part before the apostrophe.
    first_person = sum(t.lower().split("'")[0] in FIRST_PERSON for t in tokens)
    readable = n >= 10  # readability formulas are meaningless on a handful of words

    return {
        "word_count": n,
        "paragraph_count": len(split_paragraphs(text)),
        "sentence_count": len(sent_lens),
        "sentence_len_mean": statistics.mean(sent_lens) if sent_lens else 0.0,
        "sentence_len_sd": statistics.stdev(sent_lens) if len(sent_lens) > 1 else 0.0,
        "avg_word_len": statistics.mean(letters) if letters else 0.0,
        "mtld": mtld(tokens),
        "long_word_pct": per100(sum(l >= 7 for l in letters)),
        "commas_p100": per100(text.count(",")),
        "semicolons_colons_p100": per100(text.count(";") + text.count(":")),
        "dashes_p100": per100(dashes),
        "excl_quest_p100": per100(text.count("!") + text.count("?")),
        "parentheses_p100": per100(text.count("(")),
        "flesch_reading_ease": textstat.flesch_reading_ease(text) if readable else 0.0,
        "fk_grade": textstat.flesch_kincaid_grade(text) if readable else 0.0,
        "discourse_markers_p100": per100(sum(markers.values())),
        "discourse_markers_found": markers,
        "british_count": british,
        "american_count": american,
        # None when the text has no spelling variants at all, so the feature is skipped.
        "british_share": british / (british + american) if british + american else None,
        "contractions_p100": per100(sum(is_contraction(t) for t in tokens)),
        "first_person_p100": per100(first_person),
        "function_word_profile": function_word_profile(tokens),
    }
