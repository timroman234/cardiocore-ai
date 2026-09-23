"""Tests for the live-replay monitor (src/monitor.py): the data handed to the browser animation."""
import json
import re

import numpy as np
import pytest

from dsp import analyze
from monitor import monitor_html, monitor_payload
from signal_gen import synthesize_ecg


@pytest.fixture(scope="module")
def nsr():
    rec = synthesize_ecg(bpm=72, duration_s=10, fs=250, snr_db=30, seed=42)
    return rec, analyze(rec.signal, rec.fs)


def _embedded_json(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert m, "payload not embedded"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_payload_describes_the_analysed_window(nsr):
    rec, a = nsr
    p = monitor_payload(rec, a)
    assert p["fs"] == 250 and len(p["signal"]) == len(rec.signal)
    assert p["duration_s"] == pytest.approx(10.0, abs=0.01)
    pk = np.array(p["peaks_t"])
    assert len(pk) == len(a.peaks) and np.all(np.diff(pk) > 0) and pk.max() < 10.0


def test_screen_width_divides_the_window_into_whole_sweeps():
    for dur in (10, 20, 30):
        rec = synthesize_ecg(bpm=72, duration_s=dur, fs=250, seed=1)
        p = monitor_payload(rec, analyze(rec.signal, rec.fs))
        assert p["screen_s"] == pytest.approx(5.0)
        assert (p["duration_s"] / p["screen_s"]) == pytest.approx(round(p["duration_s"] / p["screen_s"]))


def test_rolling_heart_rate_matches_the_signal(nsr):
    rec, a = nsr
    bpm = monitor_payload(rec, a)["bpm_at_peak"]
    assert bpm[0] is None                                   # one beat is not a rate yet
    assert all(abs(b - 72) < 6 for b in bpm[2:])


def test_y_range_contains_every_r_peak(nsr):
    rec, a = nsr
    p = monitor_payload(rec, a)
    assert p["y_min"] < min(a.filtered) and p["y_max"] > max(a.filtered[a.peaks])


def test_html_embeds_valid_json_and_the_canvas(nsr):
    rec, a = nsr
    html = monitor_html(monitor_payload(rec, a), height=374)
    assert "<canvas" in html and "requestAnimationFrame" in html
    assert _embedded_json(html)["fs"] == 250


def test_labels_cannot_break_out_of_the_script_tag(nsr):
    rec, a = nsr
    p = monitor_payload(rec, a)
    p["label"] = "</script><img src=x onerror=alert(1)>"
    html = monitor_html(p, height=374)
    # exactly one closing script tag survives: the real one at the end
    assert html.count("</script>") == 1
    assert _embedded_json(html)["label"] == p["label"]


def test_html_stays_small_for_the_longest_window():
    rec = synthesize_ecg(bpm=72, duration_s=30, fs=500, seed=3)
    html = monitor_html(monitor_payload(rec, analyze(rec.signal, rec.fs)), height=374)
    assert len(html) < 400_000
