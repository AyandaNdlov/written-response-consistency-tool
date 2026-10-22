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
                         Flagged paragraphs ─► Follow-up questions (template or Claude) ─┘
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

### ⚠️ Reference ranges are provisional

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
- **Optional LLM mode** sends the flagged paragraphs to Anthropic's API. Enable it only if your
  organisation's data-processing agreements allow this cross-border transfer (POPIA s72).
- The question generator is instructed never to accuse, suggest AI use or judge authorship.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

Then use **Load example** to try the bundled sample pairs.

Optional LLM questions: set `ANTHROPIC_API_KEY` in your environment, or put it in
`.streamlit/secrets.toml` (git-ignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

The app uses `claude-sonnet-5-5` with server-side fallback enabled, and falls back to
template questions if the call fails.

## Calibrate the reference ranges

1. Collect genuine pairs: two texts by the same person, ideally one supervised and one remote,
   collected with consent and anonymised. Save them as `data/genuine_pairs/personX_a.txt` and
   `personX_b.txt`.
2. Run a dry run to see the proposed ranges, then save them:

```bash
python calibrate.py          # shows old vs new ranges, saves nothing
python calibrate.py --write  # saves src/reference_ranges.json
```

For each feature, `typical` = 75th percentile and `high` = 95th percentile of the
within-person differences. With few pairs these are shrunk towards the defaults (weight
n / (n + 10)), and the ranges are marked `calibrated` only after 30+ pairs.

The 3 bundled pairs in `data/genuine_pairs/` are **synthetic texts written to demonstrate the
mechanics**, not real candidate data. They use different prompts within each pair, which makes
them a harsh test.

## Evaluation plan

Three conditions, each with a supervised baseline from the same person:

| Condition | Remote text | Expected |
|---|---|---|
| Genuine-genuine | Candidate's own remote response | Mostly green |
| Genuine-AI | Fully generated response | Many amber/red |
| Genuine-AI-edited | Generated text partly rewritten by the candidate | In between; edited paragraphs green, generated ones flagged |

**Key metric: the false-flag rate on genuine pairs** (the share of features, and of candidates,
with amber/red bands when nothing is wrong). This is the fairness-critical error. Also report the
false-flag rate separately for first- and second-language English writers, and the
paragraph-level hit rate in the AI-edited condition. Because no single score exists, report band
counts per condition rather than a detection accuracy.

Run the bundled version with `python evaluate.py -v`. With the provisional ranges:

| Condition | Features outside typical | Red |
|---|---|---|
| Genuine-genuine (sample) | 1 / 17 | 0 |
| Genuine-AI (sample) | 12 / 17 | 9 |
| Genuine-AI-edited (sample) | 8 / 17 (paragraphs 2 and 3 flagged, i.e. the unedited ones) | 6 |
| 3 synthetic genuine pairs, different prompts | 11 / 49 (22%) | 2 |

The 22% false-flag rate on different-prompt genuine pairs is exactly why bands must be read as
prompts for conversation and why calibration on real data is needed.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose the repo,
   branch `main` and main file `app.py`. Under *Advanced settings* choose Python 3.12 or newer.
3. Optional: under *Advanced settings → Secrets*, add `ANTHROPIC_API_KEY = "sk-ant-..."`.
4. Deploy. `requirements.txt` is installed automatically.

## Project structure

```
app.py                  Streamlit UI
calibrate.py            Recompute reference ranges from genuine pairs
evaluate.py             Run the three evaluation conditions on the samples
src/features.py         Feature extraction (regex/plain Python; textstat for readability)
src/compare.py          Differences, banding, length checks
src/paragraphs.py       Paragraph-level analysis
src/questions.py        Template and optional Claude follow-up questions
src/report.py           DOCX report export (in memory)
src/reference_ranges.json  Typical/high within-person differences (provisional)
data/samples/           Example texts for the three conditions
data/genuine_pairs/     Synthetic genuine pairs for demonstrating calibration
tests/                  pytest tests
```
