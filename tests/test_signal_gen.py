"""Tests for the synthetic ECG generator (src/signal_gen.py)."""
import numpy as np
import pytest

from signal_gen import CLASSES, synthesize_ecg


def _rr(rec):
    return np.diff(rec.beat_times)


def test_shape_and_sampling_rate():
    rec = synthesize_ecg(bpm=72, duration_s=10, fs=250, seed=1)
    assert rec.fs == 250
    assert len(rec.signal) == len(rec.t) == 2500
    assert rec.t[1] - rec.t[0] == pytest.approx(1 / 250)


def test_mean_rate_matches_request():
    rec = synthesize_ecg(bpm=90, duration_s=20, fs=250, snr_db=np.inf, seed=2)
    assert 60 / _rr(rec).mean() == pytest.approx(90, abs=3)


def test_seed_is_reproducible():
    a = synthesize_ecg(bpm=80, seed=7)
    b = synthesize_ecg(bpm=80, seed=7)
    np.testing.assert_allclose(a.signal, b.signal)


def test_r_wave_is_dominant_deflection():
    rec = synthesize_ecg(bpm=60, duration_s=10, fs=500, snr_db=np.inf, wander_mv=0, seed=3)
    idx = np.round(rec.beat_times * rec.fs).astype(int)
    assert np.all(rec.signal[idx] > 0.8)


def test_lower_snr_means_more_noise():
    clean = synthesize_ecg(bpm=72, snr_db=np.inf, wander_mv=0, seed=4)
    noisy = synthesize_ecg(bpm=72, snr_db=5, wander_mv=0, seed=4)
    assert np.std(noisy.signal - clean.signal) > 0.1


def test_afib_is_irregular_and_pvc_has_wide_beats():
    nsr = synthesize_ecg(bpm=75, duration_s=30, rhythm="NSR", seed=5)
    afib = synthesize_ecg(bpm=75, duration_s=30, rhythm="AFib", seed=5)
    cv = lambda r: _rr(r).std() / _rr(r).mean()
    assert cv(afib) > 3 * cv(nsr)
    assert afib.label == "Atrial Fibrillation"

    pvc = synthesize_ecg(bpm=75, duration_s=30, rhythm="PVC", seed=5)
    assert pvc.label == "PVC"
    assert "V" in pvc.beat_types and "N" in pvc.beat_types
    # A PVC is premature: the RR before it is shorter than typical, and a compensatory pause follows.
    v = pvc.beat_types.index("V")
    rr = _rr(pvc)
    assert rr[v - 1] < 0.8 * np.median(rr)


@pytest.mark.parametrize("bpm,label", [(45, "Sinus Bradycardia"), (72, "NSR"), (140, "Sinus Tachycardia")])
def test_labels_follow_rate_for_sinus_rhythm(bpm, label):
    assert synthesize_ecg(bpm=bpm, seed=6).label == label
    assert label in CLASSES
