"""Headless smoke tests for the Streamlit dashboard (streamlit.testing, no browser)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _badge(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if 'class="badge"' in m.value)


def test_app_starts_without_errors_and_shows_normal_rhythm():
    at = _run()
    assert not at.exception, at.exception
    assert "NSR" in _badge(at)
    assert len(at.metric) == 6


def test_injecting_afib_changes_the_classification():
    at = _run()
    at.selectbox(key="rhythm").select("Atrial fibrillation").run()
    assert not at.exception, at.exception
    assert "ATRIAL FIBRILLATION" in _badge(at)


def test_heart_rate_slider_drives_bradycardia_and_tachycardia():
    at = _run()
    at.slider(key="bpm").set_value(45).run()
    assert "BRADYCARDIA" in _badge(at)
    at.slider(key="bpm").set_value(150).run()
    assert "TACHYCARDIA" in _badge(at)


def test_tutor_without_api_key_shows_setup_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # The app calls load_dotenv() on every run; stop it re-reading a real .env (and spending money).
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = _run()
    at.button(key="ask").click().run()
    assert not at.exception, at.exception
    assert any(".env" in e.value for e in at.error), [e.value for e in at.error]
