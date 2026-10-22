"""Paragraph-level analysis of the remote text.

Each paragraph of 40+ words is profiled and compared with the whole baseline
text. Differences are scaled by the typical within-person difference (made
more tolerant, because single paragraphs are noisier than whole texts) and
averaged into a relative difference score. Scores are for ranking paragraphs
within one response, not for judging them in isolation.
"""

from __future__ import annotations

from .compare import AMBER, GREEN, GREY, RED, SPEC_BY_KEY
from .features import extract_features, function_word_distance, split_paragraphs

MIN_PARAGRAPH_WORDS = 40
PARAGRAPH_TOLERANCE = 1.5  # paragraphs vary more than whole texts
SCORE_CAP = 3.0            # stop one extreme feature dominating the score
AMBER_SCORE = 1.0
RED_SCORE = 1.5

PARAGRAPH_FEATURES = [
    "sentence_len_mean", "avg_word_len", "long_word_pct", "commas_p100",
    "discourse_markers_p100", "contractions_p100", "first_person_p100",
    "function_word_distance",
]


def _score_band(score: float) -> str:
    if score < AMBER_SCORE:
        return GREEN
    if score < RED_SCORE:
        return AMBER
    return RED


def analyse_paragraphs(remote_text: str, baseline_features: dict, ranges: dict) -> list[dict]:
    """Return one dict per remote paragraph, in original order."""
    results = []
    for idx, para in enumerate(split_paragraphs(remote_text), start=1):
        feats = extract_features(para)
        entry = {"index": idx, "text": para, "word_count": feats["word_count"],
                 "eligible": feats["word_count"] >= MIN_PARAGRAPH_WORDS,
                 "score": None, "band": GREY, "drivers": []}
        if entry["eligible"]:
            contributions = {}
            for key in PARAGRAPH_FEATURES:
                if key == "function_word_distance":
                    diff = function_word_distance(baseline_features["function_word_profile"],
                                                  feats["function_word_profile"])
                else:
                    diff = abs(feats[key] - baseline_features[key])
                scale = ranges[key]["typical"] * PARAGRAPH_TOLERANCE
                contributions[key] = min(diff / scale, SCORE_CAP) if scale else 0.0
            score = sum(contributions.values()) / len(contributions)
            top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:2]
            entry.update(score=score, band=_score_band(score),
                         drivers=[SPEC_BY_KEY[k].label for k, v in top if v >= 1.0])
        results.append(entry)

    ranked = sorted((r for r in results if r["eligible"]), key=lambda r: r["score"], reverse=True)
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank
    return results


def flagged_paragraphs(paragraphs: list[dict], limit: int = 3) -> list[dict]:
    """Most different eligible paragraphs that are at least amber, highest first."""
    candidates = [p for p in paragraphs if p["eligible"] and p["band"] in (AMBER, RED)]
    return sorted(candidates, key=lambda p: p["score"], reverse=True)[:limit]
