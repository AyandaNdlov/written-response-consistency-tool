from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app.py")
EXAMPLES = [
    "Genuine pair (same person, both genuine)",
    "Genuine baseline vs AI-generated remote",
    "Genuine baseline vs AI-generated, partly edited",
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_app_runs_each_example(example, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.selectbox(key="example").select(example).run()
    assert at.text_area(key="baseline_text").value.strip()
    at.button[0].click().run()
    assert not at.exception
    assert any("Consistency report" in h.value for h in at.header)
    assert any("features fall outside the typical range" in m.value for m in at.markdown)


def test_app_refuses_short_text(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_area(key="baseline_text").input("Far too short to compare. " * 5)
    at.text_area(key="remote_text").input("Also too short. " * 5)
    at.button[0].click().run()
    assert not at.exception
    assert any("no comparison was made" in e.value for e in at.error)
