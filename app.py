"""Written Response Consistency Tool: Streamlit UI.

Compares a supervised baseline writing sample with a remote (unproctored)
written assessment from the same candidate. All text is processed in memory
only; nothing is written to disk (POPIA).
"""

from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.compare import (BAND_COLOURS, BAND_ICONS, GREY, compare_texts, format_value,
                         load_reference_ranges)
from src.paragraphs import analyse_paragraphs, flagged_paragraphs
from src.questions import LLM_MODEL, generate_questions, get_api_key
from src.report import DISCLAIMER, PROVISIONAL_NOTE, build_report

SAMPLES_DIR = Path(__file__).parent / "data" / "samples"
EXAMPLES = {
    "Genuine pair (same person, both genuine)": ("baseline_person1.txt", "remote_genuine_person1.txt"),
    "Genuine baseline vs AI-generated remote": ("baseline_person1.txt", "remote_ai_person1.txt"),
    "Genuine baseline vs AI-generated, partly edited": ("baseline_person1.txt", "remote_ai_edited_person1.txt"),
}

# Fixed display ranges for the radar chart, so both texts share one scale.
RADAR_FEATURES = {
    "sentence_len_mean": ("Sentence length", 5, 35),
    "sentence_len_sd": ("Sentence variation", 0, 15),
    "avg_word_len": ("Word length", 3.5, 6.0),
    "mtld": ("Vocab diversity", 30, 150),
    "long_word_pct": ("Long words %", 5, 40),
    "commas_p100": ("Commas", 0, 10),
    "discourse_markers_p100": ("Stock connectives", 0, 4),
    "contractions_p100": ("Contractions", 0, 6),
    "first_person_p100": ("First person", 0, 12),
    "flesch_reading_ease": ("Reading ease", 0, 100),
}

st.set_page_config(page_title="Written Response Consistency Tool", page_icon="📝", layout="wide")


# ---------------------------------------------------------------- helpers

def read_upload(uploaded) -> str:
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".docx"):
        from docx import Document
        return "\n\n".join(p.text for p in Document(BytesIO(data)).paragraphs if p.text.strip())
    return data.decode("utf-8", errors="replace")


def on_upload(which: str) -> None:
    uploaded = st.session_state.get(f"{which}_upload")
    if uploaded is not None:
        st.session_state[f"{which}_text"] = read_upload(uploaded)
        st.session_state.pop("analysis", None)


def on_example() -> None:
    choice = st.session_state.get("example")
    if choice in EXAMPLES:
        base_file, remote_file = EXAMPLES[choice]
        st.session_state["baseline_text"] = (SAMPLES_DIR / base_file).read_text(encoding="utf-8")
        st.session_state["remote_text"] = (SAMPLES_DIR / remote_file).read_text(encoding="utf-8")
        st.session_state.pop("analysis", None)


def radar_chart(base: dict, remote: dict) -> go.Figure:
    labels, b_vals, r_vals = [], [], []
    for key, (label, lo, hi) in RADAR_FEATURES.items():
        labels.append(label)
        b_vals.append(min(max((base[key] - lo) / (hi - lo), 0), 1))
        r_vals.append(min(max((remote[key] - lo) / (hi - lo), 0), 1))
    fig = go.Figure()
    for name, vals, colour in (("Baseline", b_vals, "#1f77b4"), ("Remote", r_vals, "#d62728")):
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name=name,
                                      fill="toself", opacity=0.55, line=dict(color=colour)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                      margin=dict(l=40, r=40, t=30, b=30), height=430,
                      legend=dict(orientation="h", y=-0.1))
    return fig


def band_style(val: str) -> str:
    colour = BAND_COLOURS.get(val.split(" ", 1)[-1])
    return f"background-color: {colour}; color: #222" if colour else ""


# ---------------------------------------------------------------- sidebar

api_key = get_api_key(st.secrets)
ranges = load_reference_ranges()

with st.sidebar:
    st.header("About this tool")
    st.markdown(
        "Compares a candidate's **supervised baseline** writing sample with their **remote** "
        "written assessment. It shows where writing patterns are similar or different, "
        "measured against how much one person's writing *typically* varies, and suggests "
        "follow-up interview questions."
    )
    st.warning(
        "**Ethical use.** This is **not** an AI detector and gives no AI score or probability. "
        "Use it only to plan a fair, structured follow-up conversation. Never use it as the "
        "sole basis for a selection decision. Tell candidates how their written work is reviewed."
    )
    st.info("**Privacy (POPIA).** Text is processed in memory only and is not stored. "
            "Remove names and ID numbers before pasting.")
    st.subheader("Question generation")
    use_llm = st.toggle("Use Claude to tailor questions", value=False, disabled=not api_key,
                        help=f"Uses {LLM_MODEL}. Requires ANTHROPIC_API_KEY.")
    if not api_key:
        st.caption("No ANTHROPIC_API_KEY found, so template questions are used.")
    elif use_llm:
        st.caption("When on, the flagged paragraphs are sent to Anthropic's API to write questions. "
                   "Make sure your organisation's data-processing agreement allows this.")
    if ranges.get("_meta", {}).get("status") != "calibrated":
        st.caption("⚠️ Reference ranges are **provisional** until calibrated (see README).")


# ---------------------------------------------------------------- main

st.title("📝 Written Response Consistency Tool")
st.caption("For assessment practitioners in graduate recruitment. Supports professional judgement; "
           "does not replace it.")
st.error(DISCLAIMER, icon="⚖️")

st.selectbox("Load example", ["(none)"] + list(EXAMPLES), key="example", on_change=on_example)

col_b, col_r = st.columns(2)
with col_b:
    st.subheader("Baseline text (supervised)")
    st.file_uploader("Upload .txt or .docx", type=["txt", "docx"], key="baseline_upload",
                     on_change=on_upload, args=("baseline",))
    st.text_area("Or paste text", key="baseline_text", height=300)
with col_r:
    st.subheader("Remote text (unproctored)")
    st.file_uploader("Upload .txt or .docx", type=["txt", "docx"], key="remote_upload",
                     on_change=on_upload, args=("remote",))
    st.text_area("Or paste text", key="remote_text", height=300)

if st.button("Compare texts", type="primary"):
    baseline = st.session_state.get("baseline_text", "")
    remote = st.session_state.get("remote_text", "")
    if not baseline.strip() or not remote.strip():
        st.warning("Please provide both a baseline and a remote text.")
    else:
        with st.spinner("Analysing writing features..."):
            result = compare_texts(baseline, remote, ranges)
            paragraphs, questions, mode = [], [], "Template"
            if result["length_status"] != "refuse":
                paragraphs = analyse_paragraphs(remote, result["baseline_features"], ranges)
                questions, mode = generate_questions(flagged_paragraphs(paragraphs), paragraphs,
                                                     use_llm=use_llm, api_key=api_key)
        st.session_state["analysis"] = dict(result=result, paragraphs=paragraphs,
                                            questions=questions, mode=mode)

analysis = st.session_state.get("analysis")
if analysis:
    result = analysis["result"]
    st.divider()
    st.header("Consistency report")

    if result["length_status"] == "refuse":
        st.error(result["length_message"])
        st.stop()
    if result["length_status"] == "warn":
        st.warning(result["length_message"])
    if result["provisional"]:
        st.info(PROVISIONAL_NOTE)

    base, remote = result["baseline_features"], result["remote_features"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features outside typical range", f"{result['n_outside']} of {result['n_assessed']}")
    m2.metric("🟠 Somewhat unusual", result["n_amber"])
    m3.metric("🔴 Notably different", result["n_red"])
    m4.metric("Words (baseline / remote)", f"{base['word_count']} / {remote['word_count']}")
    st.markdown(f"**Summary:** {result['summary']} This describes differences in writing "
                "patterns only. It is not a judgement about who wrote the text.")

    # --- feature table
    st.subheader("Feature comparison")
    table = pd.DataFrame([{
        "Feature": r["label"],
        "Unit": r["unit"],
        "Baseline": format_value(r["baseline"], r["decimals"]),
        "Remote": format_value(r["remote"], r["decimals"]),
        "Difference": format_value(r["difference"], r["decimals"]),
        "Typical / high": f"{format_value(r['typical'], r['decimals'])} / {format_value(r['high'], r['decimals'])}",
        "Band": f"{BAND_ICONS[r['band']]} {r['band']}",
    } for r in result["rows"]])
    st.dataframe(table.style.map(band_style, subset=["Band"]), hide_index=True,
                 width="stretch", height=640)
    with st.expander("What do these features mean?"):
        for r in result["rows"]:
            st.markdown(f"- **{r['label']}**: {r['explanation']}")
        st.markdown(f"- **Spelling evidence**: baseline has {base['british_count']} British / "
                    f"{base['american_count']} American variants; remote has {remote['british_count']} / "
                    f"{remote['american_count']}. Compared only when each text has at least 2.")
        if remote["discourse_markers_found"] or base["discourse_markers_found"]:
            st.markdown(f"- **Connectives found**: baseline {base['discourse_markers_found'] or 'none'}; "
                        f"remote {remote['discourse_markers_found'] or 'none'}")

    st.subheader("Writing profile")
    st.plotly_chart(radar_chart(base, remote), width="stretch")
    st.caption("Each axis is scaled to a fixed typical range (0 = low, 1 = high) so both texts share "
               "one scale. Overlapping shapes indicate similar profiles.")

    # --- paragraphs
    st.subheader("Paragraph highlights (remote text)")
    st.caption("Paragraphs of 40+ words are compared with the baseline and ranked. Darker shading = "
               "more different from the baseline. Paragraph scores are noisy; use them to choose "
               "what to ask about, not to draw conclusions.")
    for p in analysis["paragraphs"]:
        colour = BAND_COLOURS[p["band"] if p["eligible"] else GREY]
        if p["eligible"]:
            info = f"score {p['score']:.2f} · rank {p['rank']} · {BAND_ICONS[p['band']]} {p['band']}"
            if p["drivers"]:
                info += f" · largest differences: {', '.join(p['drivers'])}"
        else:
            info = "fewer than 40 words, not scored"
        st.markdown(
            f"<div style='background:{colour};color:#222;padding:0.7rem 0.9rem;border-radius:6px;"
            f"margin-bottom:0.6rem'><b>Paragraph {p['index']}</b> "
            f"<span style='font-size:0.85em'>({p['word_count']} words; {info})</span><br>"
            f"{html.escape(p['text'])}</div>", unsafe_allow_html=True)

    # --- questions
    st.subheader("Suggested follow-up questions")
    st.caption(f"Source: {analysis['mode']}. Questions probe understanding and reasoning; edit them "
               "to fit your interview guide.")
    for i, q in enumerate(analysis["questions"], start=1):
        st.markdown(f"{i}. {q}")

    # --- notes + report
    st.subheader("Practitioner notes")
    candidate_ref = st.text_input("Candidate reference (optional, avoid names)")
    notes = st.text_area("Notes (included in the report, not stored)", height=150)
    st.download_button(
        "Download report (.docx)",
        data=build_report(result, analysis["paragraphs"], analysis["questions"], analysis["mode"],
                          notes, candidate_ref),
        file_name="consistency_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
    )
