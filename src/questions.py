"""Follow-up interview questions.

Template mode needs no API key. LLM mode uses the Anthropic API and falls back
to templates on any failure. Questions focus on the candidate's reasoning and
understanding of their own response; they never refer to writing style,
authorship or AI.
"""

from __future__ import annotations

import os
import re

from .features import split_sentences

LLM_MODEL = "claude-sonnet-5-5"

LLM_SYSTEM_PROMPT = (
    "You help Industrial Psychologists and assessment practitioners prepare a structured "
    "follow-up interview after a graduate assessment. You will receive paragraphs from a "
    "candidate's written response. Generate questions that probe the candidate's understanding "
    "and reasoning about the content: ask them to explain, justify, give concrete examples, "
    "consider alternatives, or extend their ideas. Never accuse the candidate of anything, never "
    "suggest or hint at AI use or outside help, and never judge or comment on authorship or "
    "writing style. Questions must be neutral, open-ended, respectful, and answerable by any "
    "candidate who understood their own response. Return only the questions, one per line, "
    "with no numbering, headings or other text."
)

PARAGRAPH_TEMPLATES = [
    'In paragraph {n} you wrote: "{quote}" Can you walk me through how you reached that conclusion?',
    'Paragraph {n} makes the point that "{quote}" Can you give a concrete example, from your own experience, that shows this in practice?',
    'In paragraph {n} you said: "{quote}" What alternatives did you consider, and why did you settle on this approach?',
]

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
    questions = [
        PARAGRAPH_TEMPLATES[i % len(PARAGRAPH_TEMPLATES)].format(n=p["index"], quote=key_quote(p["text"]))
        for i, p in enumerate(flagged)
    ]
    return questions + GENERAL_PROBES[:n_general]


def get_api_key(secrets=None) -> str | None:
    """API key from the environment, or Streamlit secrets when given."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if secrets is not None:
        try:
            return secrets.get("ANTHROPIC_API_KEY")
        except Exception:  # no secrets.toml present
            return None
    return None


def _parse_questions(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line.endswith("?"):
            lines.append(line)
    return lines


def llm_questions(paragraphs: list[dict], api_key: str) -> list[str]:
    """Ask Claude for 3-5 tailored questions. Raises on failure."""
    import anthropic

    content = "\n\n".join(f"Paragraph {p['index']}:\n{p['text']}" for p in paragraphs)
    user_msg = (
        "Write between 3 and 5 follow-up interview questions based on the content of these "
        "paragraphs from the candidate's written response. Refer to paragraph numbers where "
        f"helpful.\n\n{content}"
    )
    client = anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=1)
    request = dict(model=LLM_MODEL, max_tokens=4000, system=LLM_SYSTEM_PROMPT,
                   messages=[{"role": "user", "content": user_msg}])
    try:
        # Server-side fallback reruns the request on another model if it is declined.
        response = client.beta.messages.create(
            **request, betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default", "output_config": {"effort": "low"}})
    except anthropic.BadRequestError:
        response = client.messages.create(**request)

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined the request.")
    text = "".join(b.text for b in response.content if b.type == "text")
    questions = _parse_questions(text)[:5]
    if len(questions) < 3:
        raise RuntimeError("The model returned too few questions.")
    return questions


def generate_questions(flagged: list[dict], all_paragraphs: list[dict],
                       use_llm: bool = False, api_key: str | None = None) -> tuple[list[str], str]:
    """Return (questions, mode) where mode describes how they were produced."""
    # With nothing flagged, base questions on the longest paragraphs instead.
    source = flagged or sorted((p for p in all_paragraphs if p["eligible"]),
                               key=lambda p: p["word_count"], reverse=True)[:2]
    if use_llm and api_key and source:
        try:
            return llm_questions(source, api_key), f"Generated by {LLM_MODEL}"
        except Exception as exc:
            return template_questions(source), f"Template (LLM unavailable: {type(exc).__name__})"
    return template_questions(source), "Template"
