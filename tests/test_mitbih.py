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


def test_label_carries_rhythm_state_from_before_the_slice():
    """Record 201 is in AFib from 0 s to ~378 s; a slice at 120 s has no marker of its own."""
    try:
        rec = load_mitbih("201", start_s=120, duration_s=10)
    except Exception as exc:
        pytest.skip(f"PhysioNet unavailable: {exc}")
    assert rec.label == "Atrial Fibrillation"
