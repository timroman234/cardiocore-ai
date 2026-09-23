"""Tests for the ML rhythm classifier (src/model.py)."""
import numpy as np
import pytest

from dsp import analyze
from model import FEATURE_NAMES, feature_vector, predict, train
from signal_gen import synthesize_ecg


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    # Small forest on a small synthetic set keeps the test quick; not saved to the real models/ dir.
    return train(n_per_class=60, seed=1, n_jobs=-1, path=None)


def test_holdout_accuracy_on_synthetic(bundle):
    assert bundle["accuracy"] >= 0.90


def test_feature_vector_handles_nan_and_bools():
    v = feature_vector({"heart_rate_bpm": 70.0, "qrs_duration_ms": float("nan"), "p_wave_present": True})
    assert v.shape == (len(FEATURE_NAMES),)
    assert not np.isnan(v).any()
    assert v[FEATURE_NAMES.index("p_wave_present")] == 1.0


@pytest.mark.parametrize("rhythm,bpm,expected", [
    ("NSR", 72, "NSR"), ("NSR", 45, "Sinus Bradycardia"), ("NSR", 140, "Sinus Tachycardia"),
    ("AFib", 90, "Atrial Fibrillation"), ("PVC", 75, "PVC"),
])
def test_predicts_fresh_synthetic_signals(bundle, rhythm, bpm, expected):
    rec = synthesize_ecg(bpm=bpm, duration_s=20, snr_db=20, rhythm=rhythm, seed=999)
    p = predict(analyze(rec.signal, rec.fs).features, bundle)
    assert p.label == expected
    assert 0.0 < p.confidence <= 1.0
    assert sum(p.probabilities.values()) == pytest.approx(1.0)
