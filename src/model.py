"""
model.py -- Machine-learning rhythm classifier (PRD section 5.1).

The idea: instead of feeding thousands of raw voltage samples to a model, we first use the DSP
engine to *extract features* (heart rate, HRV numbers, QRS width, P-wave presence, ...).  A small
Random Forest then learns which combinations of those features mean which rhythm.  This is
classic "feature engineering + classical ML" -- fast, tiny, and (unlike a deep net) explainable.

Random Forest in one paragraph: train many decision trees, each on a random resample of the data
and a random subset of features; every tree votes and the vote share becomes the model's
confidence.  Averaging many noisy-but-different trees cancels their individual mistakes.

Training data: synthetic windows from signal_gen.py, with random heart rate, noise and
recording length, pushed through the *same* DSP pipeline used at inference time so that training
and prediction features match.  Important caveat for teaching: a model trained only on
synthetic data can look excellent on synthetic data and still stumble on real patients -- Mode B
in the app lets students see that "domain gap" for themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from dsp import analyze
from signal_gen import CLASSES, synthesize_ecg

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "rhythm_rf.joblib"

# Order matters: this is the column order of the training matrix.
FEATURE_NAMES = [
    "heart_rate_bpm",       # rate: separates bradycardia / normal / tachycardia
    "sdnn_ms",              # overall RR variability
    "rmssd_ms",             # beat-to-beat variability
    "pnn50",                # % of successive RR differences > 50 ms
    "rr_cv",                # SDNN / mean RR: variability relative to rate
    "rr_min_max_ratio",     # shortest RR / longest RR (1.0 = perfectly regular)
    "rmssd_norm",           # RMSSD / mean RR
    "qrs_duration_ms",      # median QRS width
    "wide_qrs_fraction",    # share of beats with QRS >= 120 ms (PVC signature)
    "p_wave_present",       # 1 if a clear P wave precedes the QRS (absent in AFib)
    "p_wave_amplitude_mv",  # how big that P wave is
]


def feature_vector(features: dict) -> np.ndarray:
    """Turn the DSP feature dict into the fixed-order numeric row the model expects."""
    row = []
    for name in FEATURE_NAMES:
        v = features.get(name, 0.0)
        v = 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        row.append(v)
    return np.array(row)


@dataclass
class Prediction:
    label: str
    confidence: float             # fraction of trees voting for `label`
    probabilities: dict[str, float]


# ----------------------------------------------------------------------------------------------
# Synthetic training set
# ----------------------------------------------------------------------------------------------
# (rhythm passed to the generator, heart-rate range in bpm) for each class.  Sinus classes are
# defined by rate; AFib and PVC can occur at any ordinary underlying rate.
_RECIPES = {
    "NSR": ("NSR", (60, 100)),
    "Sinus Bradycardia": ("NSR", (40, 59)),
    "Sinus Tachycardia": ("NSR", (101, 180)),
    "Atrial Fibrillation": ("AFib", (50, 170)),
    "PVC": ("PVC", (50, 110)),
}


# How often PVCs occur is itself variable in patients: an isolated one now and then, every third
# beat (trigeminy), or every other beat (bigeminy).  Training on only "occasional" PVCs made the
# model mistake heavy ectopy (e.g. MIT-BIH record 208: >50% ectopic beats) for atrial fibrillation.
_PVC_BURDENS = [(2, 3), (2, 4), (3, 6), (4, 9)]   # ranges of "beats between PVCs" (lo, hi)

# Likewise the P wave: textbook amplitude, weakened, or invisible.  Real single-lead recordings
# (e.g. MIT-BIH record 208) often show no detectable P wave even in non-AF rhythms.  Without this the
# model learned the shortcut "no P wave + irregular = AFib" and mislabelled heavy ectopy.  What
# separates PVC from AFib is the fraction of WIDE beats, which the forest must now rely on.
_PVC_P_SCALES = [1.0, 1.0, 0.4, 0.0]


def _one_window(cls: str, seed: int) -> np.ndarray:
    """Generate one randomised labelled window and return its feature row."""
    rng = np.random.default_rng(seed)
    rhythm, (lo, hi) = _RECIPES[cls]
    pvc_interval = _PVC_BURDENS[int(rng.integers(len(_PVC_BURDENS)))]
    p_scale = _PVC_P_SCALES[int(rng.integers(len(_PVC_P_SCALES)))] if rhythm == "PVC" else 1.0
    rec = synthesize_ecg(
        bpm=float(rng.uniform(lo, hi)),
        duration_s=float(rng.uniform(10, 20)),
        fs=float(rng.choice([250, 250, 500])),
        snr_db=float(rng.uniform(8, 35)),        # from very noisy to almost clean
        wander_mv=float(rng.uniform(0.0, 0.3)),
        rhythm=rhythm,
        seed=int(rng.integers(0, 2**31 - 1)),
        pvc_interval=pvc_interval,
        p_scale=p_scale,
    )
    return feature_vector(analyze(rec.signal, rec.fs).features)


def build_training_set(n_per_class: int = 300, seed: int = 0, n_jobs: int = -1):
    """Balanced synthetic dataset: X (windows x features), y (class names)."""
    jobs = [(cls, seed * 1_000_003 + i * 97 + k) for k, cls in enumerate(CLASSES) for i in range(n_per_class)]
    rows = Parallel(n_jobs=n_jobs)(delayed(_one_window)(c, s) for c, s in jobs)
    return np.vstack(rows), np.array([c for c, _ in jobs])


def train(n_per_class: int = 300, seed: int = 0, n_jobs: int = -1, path: Path | None = MODEL_PATH) -> dict:
    """
    Train the forest, evaluate on a held-out 25%, and (optionally) save it.

    Hold-out evaluation matters: accuracy on the data a model was trained on says nothing about
    how it will do on new signals, so we score on windows the trees never saw.
    """
    X, y = build_training_set(n_per_class, seed, n_jobs)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    bundle = dict(
        clf=clf,
        features=FEATURE_NAMES,
        classes=list(clf.classes_),
        accuracy=float(accuracy_score(y_te, pred)),
        confusion=confusion_matrix(y_te, pred, labels=clf.classes_).tolist(),
        importances=dict(zip(FEATURE_NAMES, map(float, clf.feature_importances_))),
        n_train=len(X_tr),
    )
    if path is not None:
        path.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, path)
    return bundle


def load_or_train(path: Path = MODEL_PATH) -> dict:
    """Load the cached model, training it on first use (about a minute; cached after that)."""
    if path.exists():
        return joblib.load(path)
    return train(path=path)


def predict(features: dict, bundle: dict | None = None) -> Prediction:
    """Classify one analysed window."""
    bundle = bundle or load_or_train()
    clf = bundle["clf"]
    proba = clf.predict_proba(feature_vector(features).reshape(1, -1))[0]
    probs = {c: float(p) for c, p in zip(clf.classes_, proba)}
    label = max(probs, key=probs.get)
    return Prediction(label=label, confidence=probs[label], probabilities=probs)


if __name__ == "__main__":  # `uv run python src/model.py` pre-trains the model
    b = train()
    print(f"hold-out accuracy: {b['accuracy']:.3f}  (saved to {MODEL_PATH})")
