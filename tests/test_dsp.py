"""Tests for the DSP engine (src/dsp.py)."""
import numpy as np
import pytest

from dsp import analyze, bandpass, pan_tompkins, score_detection
from signal_gen import synthesize_ecg


def test_bandpass_removes_drift_and_mains_keeps_heartbeat_band():
    fs = 250
    t = np.arange(0, 20, 1 / fs)
    drift = np.sin(2 * np.pi * 0.3 * t)       # below 0.5 Hz  -> should be removed
    mains = np.sin(2 * np.pi * 60 * t)        # above 45 Hz   -> should be removed
    keep = np.sin(2 * np.pi * 10 * t)         # inside band   -> should survive
    out = bandpass(drift + mains + keep, fs)
    core = slice(int(2 * fs), int(-2 * fs))   # ignore filter edge effects
    np.testing.assert_allclose(out[core], keep[core], atol=0.1)


def test_bandpass_rejects_cutoff_above_nyquist():
    with pytest.raises(ValueError):
        bandpass(np.zeros(100), fs=80, low=0.5, high=45)


@pytest.mark.parametrize("bpm", [50, 72, 110, 160])
def test_heart_rate_recovered(bpm):
    rec = synthesize_ecg(bpm=bpm, duration_s=20, fs=250, snr_db=25, seed=11)
    a = analyze(rec.signal, rec.fs)
    assert a.features["heart_rate_bpm"] == pytest.approx(bpm, abs=2.5)


@pytest.mark.parametrize("rhythm", ["NSR", "AFib", "PVC"])
def test_pan_tompkins_finds_almost_every_beat_at_20db(rhythm):
    rec = synthesize_ecg(bpm=75, duration_s=30, fs=250, snr_db=20, rhythm=rhythm, seed=12)
    a = analyze(rec.signal, rec.fs)
    s = score_detection(a.peaks / rec.fs, rec.beat_times, duration_s=30)
    assert s["sensitivity"] >= 0.98 and s["ppv"] >= 0.98, s


def test_detection_degrades_gracefully_at_5db():
    rec = synthesize_ecg(bpm=75, duration_s=30, fs=250, snr_db=5, seed=13)
    a = analyze(rec.signal, rec.fs)
    s = score_detection(a.peaks / rec.fs, rec.beat_times, duration_s=30)
    assert s["sensitivity"] >= 0.9 and s["ppv"] >= 0.9, s


def test_hrv_separates_regular_from_afib():
    nsr = analyze(synthesize_ecg(bpm=75, duration_s=30, seed=14).signal, 250).features
    af = analyze(synthesize_ecg(bpm=75, duration_s=30, rhythm="AFib", seed=14).signal, 250).features
    assert af["rr_cv"] > 3 * nsr["rr_cv"]
    assert af["rmssd_ms"] > 3 * nsr["rmssd_ms"]


def test_pvc_has_wide_qrs_fraction_and_normal_beats_are_narrow():
    nsr = analyze(synthesize_ecg(bpm=75, duration_s=30, snr_db=25, seed=15).signal, 250).features
    pvc = analyze(synthesize_ecg(bpm=75, duration_s=30, snr_db=25, rhythm="PVC", seed=15).signal, 250).features
    assert 60 <= nsr["qrs_duration_ms"] <= 115
    assert nsr["wide_qrs_fraction"] < 0.05
    assert pvc["wide_qrs_fraction"] > 0.08


def test_p_wave_present_in_nsr_absent_in_afib():
    nsr = analyze(synthesize_ecg(bpm=75, duration_s=30, snr_db=25, seed=16).signal, 250).features
    af = analyze(synthesize_ecg(bpm=75, duration_s=30, snr_db=25, rhythm="AFib", seed=16).signal, 250).features
    assert nsr["p_wave_present"] is True
    assert af["p_wave_present"] is False


def test_interval_estimates_are_physiologic_for_nsr():
    f = analyze(synthesize_ecg(bpm=70, duration_s=30, snr_db=25, seed=17).signal, 250).features
    assert 100 <= f["pr_interval_ms"] <= 230
    assert 300 <= f["qt_interval_ms"] <= 480


def test_pipeline_stage_shapes_match_input():
    rec = synthesize_ecg(bpm=72, duration_s=10, seed=18)
    pt = pan_tompkins(analyze(rec.signal, rec.fs).filtered, rec.fs)
    for stage in (pt.band, pt.derivative, pt.squared, pt.integrated):
        assert stage.shape == rec.signal.shape


def test_score_detection_counts():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    det = np.array([1.01, 2.0, 3.5, 4.02])
    s = score_detection(det, truth, margin_s=0.0, duration_s=5)
    assert (s["tp"], s["fp"], s["fn"]) == (3, 1, 1)
