"""Compare two feature sets against typical within-person variation.

The tool never flags a raw difference. Each feature's absolute difference is
compared with how much the same person's writing typically varies between two
occasions (reference_ranges.json), and banded:

    difference <= typical          -> Within typical range  (green)
    typical < difference <= high   -> Somewhat unusual      (amber)
    difference > high              -> Notably different     (red)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .features import extract_features, function_word_distance

REFERENCE_PATH = Path(__file__).with_name("reference_ranges.json")

# Length limits: below MIN_WORDS_REFUSE no comparison is made; below
# MIN_WORDS_WARN the comparison runs but is labelled unreliable.
MIN_WORDS_REFUSE = 80
MIN_WORDS_WARN = 150
MIN_SPELLING_VARIANTS = 2  # per text, before spelling convention is compared

# Band names double as the labels shown in the UI and the report.
GREEN = "Within typical range"
AMBER = "Somewhat unusual"
RED = "Notably different"
GREY = "Not enough evidence"

# Soft fills for the table, paragraph cards and DOCX shading (green, yellow, pink, grey).
BAND_COLOURS = {GREEN: "#dcefd6", AMBER: "#fbefb0", RED: "#ffcfe0", GREY: "#ebe7e0"}
BAND_ICONS = {GREEN: "🟢", AMBER: "🟠", RED: "🔴", GREY: "⚪"}


@dataclass(frozen=True)
class FeatureSpec:
    """How one feature is labelled, formatted and explained to the practitioner."""

    key: str
    label: str
    unit: str
    decimals: int
    explanation: str
    pairwise: bool = False  # measured between the texts, not per text


# Every compared feature, in the order it appears in the table and the report.
FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec("sentence_len_mean", "Average sentence length", "words", 1,
                "How long sentences are on average. Remote, edited writing often has longer sentences."),
    FeatureSpec("sentence_len_sd", "Sentence length variation", "words (SD)", 1,
                "How much sentence length varies. Most people mix short and long sentences; very even lengths are less typical of spontaneous writing."),
    FeatureSpec("avg_word_len", "Average word length", "letters", 2,
                "Average number of letters per word, a rough indicator of vocabulary formality."),
    FeatureSpec("mtld", "Vocabulary diversity (MTLD)", "score", 0,
                "How varied the vocabulary is, adjusted for text length. Higher means fewer repeated words."),
    FeatureSpec("long_word_pct", "Long words (7+ letters)", "% of words", 1,
                "Share of longer words. Tends to rise with formal or academic register."),
    FeatureSpec("function_word_distance", "Function-word profile", "cosine distance", 3,
                "How differently the two texts use small words such as 'the', 'of', 'but' and 'which'. People use these habitually, so the profile is fairly stable within a person.",
                pairwise=True),
    FeatureSpec("commas_p100", "Commas", "per 100 words", 2,
                "Comma use reflects personal punctuation habits and sentence structure."),
    FeatureSpec("semicolons_colons_p100", "Semicolons and colons", "per 100 words", 2,
                "Many writers rarely use these; a sudden increase or drop is worth noting."),
    FeatureSpec("dashes_p100", "Dashes", "per 100 words", 2,
                "Use of dashes as punctuation (not hyphens inside words), which tends to be a personal habit."),
    FeatureSpec("excl_quest_p100", "Exclamation and question marks", "per 100 words", 2,
                "Rhetorical questions and exclamations reflect tone and personal style."),
    FeatureSpec("parentheses_p100", "Parentheses", "per 100 words", 2,
                "Use of brackets for asides or clarifications."),
    FeatureSpec("flesch_reading_ease", "Reading ease (Flesch)", "0-100", 1,
                "Higher = easier to read. Influenced by sentence length and word length together."),
    FeatureSpec("fk_grade", "Reading grade (Flesch-Kincaid)", "grade level", 1,
                "Approximate school grade needed to read the text easily."),
    FeatureSpec("discourse_markers_p100", "Stock connectives and phrases", "per 100 words", 2,
                "Rate of formal connectives and stock phrases (e.g. 'furthermore', 'moreover', 'it is important to note', 'crucial'). These are also common in templates, coaching material and edited text."),
    FeatureSpec("british_share", "British spelling share", "proportion", 2,
                "Proportion of British variants (organise, behaviour, colour) among words with British/American alternatives. South African writers usually use British spelling, but spell-checker settings can change this."),
    FeatureSpec("contractions_p100", "Contractions", "per 100 words", 2,
                "Use of forms such as 'I'd', 'don't' and 'it's'. Reflects how informal the writing is."),
    FeatureSpec("first_person_p100", "First-person pronouns", "per 100 words", 2,
                "Use of 'I', 'my', 'we' and similar. Reflects how personal and experience-based the response is."),
]
SPEC_BY_KEY = {s.key: s for s in FEATURE_SPECS}


def load_reference_ranges(path: Path | str = REFERENCE_PATH) -> dict:
    """Load the typical and high within-person differences for each feature."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def is_provisional(ranges: dict) -> bool:
    """True until calibrate.py has marked the ranges as calibrated on real data."""
    return ranges.get("_meta", {}).get("status", "provisional") != "calibrated"


def check_lengths(baseline_words: int, remote_words: int) -> tuple[str, str]:
    """Return (status, message) where status is 'ok', 'warn' or 'refuse'."""
    shortest = min(baseline_words, remote_words)
    if shortest < MIN_WORDS_REFUSE:
        return ("refuse",
                f"At least one text has fewer than {MIN_WORDS_REFUSE} words "
                f"(baseline: {baseline_words}, remote: {remote_words}). "
                "Writing features are too unstable at this length to compare, so no comparison was made.")
    if shortest < MIN_WORDS_WARN:
        return ("warn",
                f"At least one text has fewer than {MIN_WORDS_WARN} words "
                f"(baseline: {baseline_words}, remote: {remote_words}). "
                "Features from short texts are unstable, so treat this comparison as unreliable.")
    return ("ok", "")


def feature_differences(base: dict, remote: dict) -> dict[str, float | None]:
    """Absolute difference per compared feature (None when not measurable)."""
    diffs: dict[str, float | None] = {}
    for spec in FEATURE_SPECS:
        # The function-word profile is a single distance between the two texts.
        if spec.key == "function_word_distance":
            diffs[spec.key] = function_word_distance(
                base["function_word_profile"], remote["function_word_profile"])
        # Spelling share is only compared when both texts have enough spelling variants.
        elif spec.key == "british_share":
            enough = all(f["british_count"] + f["american_count"] >= MIN_SPELLING_VARIANTS
                         for f in (base, remote))
            diffs[spec.key] = abs(base["british_share"] - remote["british_share"]) if enough else None
        else:
            diffs[spec.key] = abs(base[spec.key] - remote[spec.key])
    return diffs


def band_for(diff: float | None, typical: float, high: float) -> str:
    """Place a difference in a band. A missing difference means not enough evidence."""
    if diff is None:
        return GREY
    if diff <= typical:
        return GREEN
    if diff <= high:
        return AMBER
    return RED


def compare_texts(baseline_text: str, remote_text: str, ranges: dict | None = None) -> dict:
    """Run the full feature comparison. Returns a plain dict for the UI and report."""
    ranges = ranges or load_reference_ranges()
    base = extract_features(baseline_text)
    remote = extract_features(remote_text)
    status, message = check_lengths(base["word_count"], remote["word_count"])
    result = {
        "baseline_features": base,
        "remote_features": remote,
        "length_status": status,
        "length_message": message,
        "provisional": is_provisional(ranges),
        "rows": [],
    }
    if status == "refuse":
        return result

    diffs = feature_differences(base, remote)
    for spec in FEATURE_SPECS:
        ref = ranges[spec.key]
        diff = diffs[spec.key]
        # Pairwise features have no per-text value, only a difference.
        if spec.pairwise:
            b_val, r_val = None, None
        else:
            b_val, r_val = base[spec.key], remote[spec.key]
        result["rows"].append({
            "key": spec.key,
            "label": spec.label,
            "unit": spec.unit,
            "decimals": spec.decimals,
            "explanation": spec.explanation,
            "baseline": b_val,
            "remote": r_val,
            "difference": diff,
            "typical": ref["typical"],
            "high": ref["high"],
            "band": band_for(diff, ref["typical"], ref["high"]),
        })

    # Summarise as counts only. Grey (unmeasurable) features are left out of the total.
    assessed = [r for r in result["rows"] if r["band"] != GREY]
    outside = [r for r in assessed if r["band"] in (AMBER, RED)]
    result["n_assessed"] = len(assessed)
    result["n_outside"] = len(outside)
    result["n_amber"] = sum(r["band"] == AMBER for r in assessed)
    result["n_red"] = sum(r["band"] == RED for r in assessed)
    result["summary"] = (
        f"{len(outside)} of {len(assessed)} features fall outside the typical range "
        f"({result['n_amber']} somewhat unusual, {result['n_red']} notably different)."
    )
    return result


def format_value(value: float | None, decimals: int) -> str:
    """Format a number for display, using a hyphen when there is no value."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"
