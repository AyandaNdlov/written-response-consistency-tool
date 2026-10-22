# Written Response Consistency Tool

A Streamlit app for Industrial Psychologists and assessment practitioners in graduate
recruitment. It compares two texts from the same candidate:

1. **Baseline**: a short writing sample completed under supervised conditions (e.g. at an assessment centre)
2. **Remote**: a written assessment completed remotely and unproctored

It produces a **consistency report** showing where writing patterns are similar or different,
highlights the paragraphs that differ most from the baseline, and suggests **follow-up
interview questions**. The practitioner makes the final judgement.

> **This is not an AI detector.** It never states or implies that a candidate did or did not
> use AI, and it produces no "AI score" or probability.
>
> *This report does not indicate whether AI was used. Differences in writing can have many
> legitimate causes, including time pressure, topic, language background, editing tools such as
> spell-checkers, and accessibility accommodations. Use this information only to guide a
> structured follow-up conversation. The final judgement rests with the practitioner.*

---

## The problem

Graduate recruiters increasingly use remote, unproctored written assessments. Generative AI
makes it easy to submit polished text that does not reflect a candidate's own writing or
reasoning. Detection tools are unreliable and biased against second-language writers, so
using them to make decisions is unfair and hard to defend. Practitioners need something
better: a **transparent, explainable comparison** with the candidate's own supervised writing,
used to plan a **fair follow-up conversation** rather than to make an accusation.

## Audience

- Industrial Psychologists and psychometrists running graduate assessment centres
- Talent acquisition practitioners who review written exercises
- **Not** for automated screening or for use without a trained practitioner

## How it works

```
 Baseline text ─┐
                ├─► Feature extraction ─► Compare each difference with typical ─► Consistency report
 Remote text ───┘   (17 writing features)  within-person variation (bands)        + traffic-light table
                                                                                  + radar chart
 Remote text ─► Split into paragraphs ─► Distance of each paragraph from baseline ─► Ranked highlights
                                                                                         │
                         Flagged paragraphs ─► Follow-up questions (templates) ─────────┘
                                                                                         ▼
                                                            Practitioner notes ─► DOCX report download
```

## Features and why each matters

| Feature | What it measures | Why it matters |
|---|---|---|
| Average sentence length | Words per sentence | Edited or generated prose often has longer sentences |
| Sentence length variation (SD) | Spread of sentence lengths | Spontaneous writing mixes short and long sentences; very even lengths are less typical |
| Average word length | Letters per word | Rough indicator of vocabulary formality |
| Vocabulary diversity (MTLD) | Lexical diversity, length-adjusted (own implementation) | Sudden jumps in vocabulary range are worth exploring |
| Long words (7+ letters) | % of words | Rises with formal or academic register |
| Function-word profile | Cosine distance between relative frequencies of ~60 function words (Burrows-style) | Function words are used habitually and are a classic stylometric signal of individual style |
| Commas, semicolons/colons, dashes, !/?, parentheses | Per 100 words | Punctuation habits are fairly stable within a person |
| Flesch Reading Ease, Flesch-Kincaid Grade | `textstat` | Overall readability and register |
| Stock connectives and phrases | Per 100 words ("furthermore", "moreover", "it is important to note", "crucial", "delve", "in today's", etc.) | Common in templated, coached or generated text, but also in good formal writing |
| British spelling share | British vs American variants (organise/organize, behaviour/behavior, analyse/analyze, centre/center, programme/program, -ise/-ize) | South African candidates usually use British spelling; compared only when each text has at least 2 variants |
| Contractions | Per 100 words | Informality of register |
| First-person pronouns | Per 100 words | How personal and experience-based the response is |
| Word and paragraph counts | Shown for context | Used for the minimum-length checks |

## Banding method

People's writing naturally varies, especially between a timed supervised sample and a
polished remote response, so **raw differences are never flagged**. For each feature:

1. Compute the absolute difference between baseline and remote.
2. Compare it with the **typical** and **high** within-person differences in
   [`src/reference_ranges.json`](src/reference_ranges.json):
   - difference ≤ typical → 🟢 **Within typical range**
   - typical < difference ≤ high → 🟠 **Somewhat unusual**
   - difference > high → 🔴 **Notably different**
3. Summarise as a count, e.g. *"3 of 17 features fall outside the typical range"*. No single
   score is computed.

**Length checks:** under 150 words in either text, the app warns that the comparison is
unreliable. Under 80 words, it refuses to compare.

**Paragraph analysis:** every remote paragraph of 40+ words is profiled on 8 features. Each
difference from the baseline is divided by 1.5 × the typical difference (paragraphs are noisier
than whole texts) and capped at 3. The mean of these gives a relative score: below 1.0 is
green, 1.0 to 1.5 amber, and above 1.5 red. Paragraphs are ranked and shaded in the UI.

### Reference ranges are provisional

The values shipped in `reference_ranges.json` are **judgement-based provisional defaults** and
have **not** been calibrated on real candidate data. The UI and the report both say so. Replace
them by running `calibrate.py` on genuine pairs (see below).

## Limitations

- **Not an AI detector.** Large differences have many legitimate causes, and close similarity
  proves nothing either.
- **Short texts:** stylometric features are unstable under ~150 words, and paragraph-level scores are
  noisier still. MTLD in particular can swing widely on short texts.
- **Second-language writers** may write very differently under time pressure than with time to
  edit, and may rely more on tools. This is legitimate.
- **Grammarly and spell-checkers** change spelling, punctuation and phrasing. A US-English
  spell-checker can switch spelling conventions on its own.
- **Topic and genre effects:** if the baseline and remote prompts differ, features (especially
  the function-word profile and first-person use) differ for reasons unrelated to authorship.
- **Provisional ranges** until calibrated on a representative sample.
- **Accessibility accommodations** such as dictation software or extra time change writing style.

## Ethical use and POPIA

- Use the output **only** to plan a structured, consistent follow-up conversation, and ask every
  candidate in a cohort comparable questions. Never use it as the sole basis for a decision.
- Tell candidates in advance that remote written work may be discussed at interview.
- **POPIA:** text is processed **in memory only**. The app writes nothing to disk and keeps no
  logs, and Streamlit usage statistics are disabled. Remove names and ID numbers before pasting.
  The downloaded report is the practitioner's responsibility to store securely.
- Follow-up questions come from fixed templates that never accuse, suggest AI use or judge
  authorship. No text is sent to any outside service.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

Then use **Load example** to try the bundled sample pairs, or upload the extra samples in
`data/samples/person2_*.txt`.


