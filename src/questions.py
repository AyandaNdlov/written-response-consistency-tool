"""Follow-up interview questions.

Questions are built from templates that quote the most different paragraphs.
They focus on the candidate's reasoning and understanding of their own
response; they never refer to writing style, authorship or AI.
"""

from __future__ import annotations

from .features import split_sentences

# Used in turn for each flagged paragraph. {n} is the paragraph number, {quote} its opening.
PARAGRAPH_TEMPLATES = [
    'In paragraph {n} you wrote: "{quote}" Can you walk me through how you reached that conclusion?',
    'Paragraph {n} makes the point that "{quote}" Can you give a concrete example, from your own experience, that shows this in practice?',
    'In paragraph {n} you said: "{quote}" What alternatives did you consider, and why did you settle on this approach?',
]

# General questions added after the paragraph questions, suitable for any response.
GENERAL_PROBES = [
    "What sources or experiences informed this response?",
    "Which part of the response did you find hardest to write, and why?",
    "If you had another hour, what would you add or change in your response?",
]


def key_quote(paragraph: str, max_words: int = 22) -> str:
    """First sentence of a paragraph, shortened for quoting."""
    sentences = split_sentences(paragraph)
    first = sentences[0] if sentences else paragraph
    words = first.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(",;:") + "…"
    return first


def template_questions(flagged: list[dict], n_general: int = 2) -> list[str]:
    """One templated question per flagged paragraph, then a few general probes."""
    questions = [
        PARAGRAPH_TEMPLATES[i % len(PARAGRAPH_TEMPLATES)].format(n=p["index"], quote=key_quote(p["text"]))
        for i, p in enumerate(flagged)
    ]
    return questions + GENERAL_PROBES[:n_general]


def generate_questions(flagged: list[dict], all_paragraphs: list[dict]) -> tuple[list[str], str]:
    """Return (questions, mode) where mode describes how they were produced."""
    # With nothing flagged, base questions on the longest paragraphs instead.
    source = flagged or sorted((p for p in all_paragraphs if p["eligible"]),
                               key=lambda p: p["word_count"], reverse=True)[:2]
    return template_questions(source), "Template"
