"""Real-data check against MIT-BIH ground truth (needs internet the first time; cached afterwards)."""
import pytest

from dsp import analyze, score_detection
from signal_gen import load_mitbih

pytestmark = pytest.mark.network


@pytest.mark.parametrize("record", ["100", "119", "201", "208"])
def test_detector_agrees_with_cardiologist_annotations(record):
    try:
        rec = load_mitbih(record, duration_s=30)
    except Exception as exc:  # offline / PhysioNet unreachable
        pytest.skip(f"PhysioNet unavailable: {exc}")
    a = analyze(rec.signal, rec.fs)
    s = score_detection(a.peaks / rec.fs, rec.beat_times, duration_s=30)
    assert s["sensitivity"] >= 0.95 and s["ppv"] >= 0.95, s
    assert rec.source == "mitbih" and rec.fs == 360.0
