"""
signal_gen.py -- Where the EKG time series come from.

Two sources feed the rest of the app (PRD section 3):

  Mode A  synthesize_ecg()  A mathematical model: every heartbeat is a sum of five Gaussian
                            bumps (the P, Q, R, S and T waves), plus noise and baseline wander.
  Mode B  load_mitbih()     Real patient recordings from the MIT-BIH Arrhythmia Database,
                            fetched from PhysioNet with the `wfdb` package and cached locally.

Both return an `EcgRecord`, so the DSP engine never needs to know which one it was given.

----------------------------------------------------------------------------------------------
Time-series vocabulary used throughout this file
----------------------------------------------------------------------------------------------
* A continuous voltage v(t) is *sampled* every 1/Fs seconds, giving a discrete array x[n].
  Fs is the sampling rate (250 or 500 Hz here; MIT-BIH uses 360 Hz).
* Nyquist: a sampled signal can only represent frequencies below Fs/2. At Fs = 250 Hz that is
  125 Hz, comfortably above the ~45 Hz of useful ECG content.
* SNR (signal-to-noise ratio) in decibels:  SNR_dB = 10 * log10(P_signal / P_noise)
  where P is mean power (mean of the squared samples).  +10 dB == 10x more signal power than noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The rhythm classes the ML model predicts (PRD section 5.1).  The PRD lumps bradycardia and
# tachycardia together; we keep them separate because they are distinguished by rate alone.
CLASSES = [
    "NSR",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "Atrial Fibrillation",
    "PVC",
]

# Rhythm names the synthetic generator can be asked for.
RHYTHMS = ("NSR", "AFib", "PVC")


@dataclass
class EcgRecord:
    """A single-lead ECG window plus whatever ground truth we know about it."""

    t: np.ndarray            # time axis in seconds
    signal: np.ndarray       # voltage in millivolts (mV)
    fs: float                # sampling rate in Hz
    beat_times: np.ndarray   # true R-peak times in seconds (ground truth for scoring detection)
    beat_types: list[str]    # "N" = normal beat, "V" = premature ventricular beat, others (MIT-BIH)
    label: str               # one of CLASSES
    source: str = "synthetic"
    description: str = ""
    params: dict = field(default_factory=dict)


def rate_label(bpm: float) -> str:
    """Sinus rhythm is called bradycardia below 60 bpm and tachycardia above 100 bpm."""
    if bpm < 60:
        return "Sinus Bradycardia"
    if bpm > 100:
        return "Sinus Tachycardia"
    return "NSR"


# ----------------------------------------------------------------------------------------------
# Mode A: Gaussian-superposition ECG model
# ----------------------------------------------------------------------------------------------
#
# Each deflection of one heartbeat is modelled as a Gaussian bump:
#
#       f(t) = A * exp( -(t - mu)^2 / (2 * sigma^2) )
#
#   A      amplitude in mV (negative = downward deflection)
#   mu     centre of the bump, measured relative to the R peak in seconds
#   sigma  width: ~95% of the bump lies within mu +/- 2*sigma
#
# One beat is the sum of five bumps:   ECG_beat(t) = f_P + f_Q + f_R + f_S + f_T
# and the whole recording is the sum of many beats, one every RR seconds:
#
#       ECG(t) = sum over beats k of  ECG_beat(t - t_k)
#
# This is a simplification of real cardiac electrophysiology, but it reproduces the textbook
# morphology and lets us control every parameter exactly.
#
#               (amplitude mV, offset from R in s, sigma in s)
_NORMAL_WAVES = {
    "P": (0.15, -0.16, 0.025),   # atrial depolarisation: small, slow bump before the QRS
    "Q": (-0.12, -0.028, 0.010),  # first downward spike of the QRS complex
    "R": (1.00, 0.000, 0.011),   # tall narrow spike: ventricular depolarisation
    "S": (-0.22, 0.030, 0.012),  # downward spike after R
    "T": (0.30, 0.240, 0.050),   # ventricular repolarisation: broad bump after the QRS
}

# A premature ventricular contraction starts in the ventricle instead of the sinus node, so the
# impulse spreads slowly through muscle rather than the fast conduction system: the QRS is WIDE
# (large sigma), TALL, has NO P wave, and the T wave points the opposite way.
_PVC_WAVES = {
    "Q": (-0.05, -0.050, 0.020),
    "R": (1.30, 0.000, 0.035),
    "S": (-0.50, 0.070, 0.030),
    "T": (-0.45, 0.280, 0.070),
}


def _gaussian(t: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    """f(t) = A * exp(-(t - mu)^2 / (2 sigma^2))."""
    return amp * np.exp(-((t - mu) ** 2) / (2.0 * sigma**2))


def _beat_schedule(rhythm: str, bpm: float, duration_s: float, rng: np.random.Generator):
    """
    Decide *when* each beat happens.  Returns (times, types, rr_before), where rr_before[k] is the
    interval that preceded beat k.  The rhythm is defined by how these intervals behave:

      NSR   regular: RR ~ 60/bpm with ~2% beat-to-beat jitter plus a gentle 0.25 Hz swing
            (respiratory sinus arrhythmia -- heart rate rises slightly on inhalation).
      AFib  "irregularly irregular": RR is drawn from a wide log-normal distribution (CV ~ 25%)
            with no pattern.  The atria quiver instead of contracting, so the P wave is absent.
      PVC   NSR with an occasional early beat.  The PVC arrives at ~60% of the normal RR and is
            followed by a compensatory pause (~140%) so the *next* normal beat stays on the
            sinus node's original schedule: 0.6 + 1.4 = 2.0 normal intervals.
    """
    mean_rr = 60.0 / bpm
    t = float(rng.uniform(0.3, 0.6))          # time of the first R peak
    times, types, rr_before = [t], ["N"], [mean_rr]
    until_pvc = int(rng.integers(3, 6))       # beats until the next premature one
    owe_pause = False

    while t < duration_s + 1.0:               # generate one extra second so the window edge is filled
        if rhythm == "AFib":
            sigma = 0.25                      # log-normal shape parameter ~ CV of 25%
            rr = mean_rr * rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma)  # mean-preserving
            rr = float(np.clip(rr, 0.3 * mean_rr, 2.2 * mean_rr))
            kind = "N"
        else:
            rr = mean_rr * (1.0 + 0.02 * rng.standard_normal() + 0.03 * np.sin(2 * np.pi * 0.25 * t))
            kind = "N"
            if rhythm == "PVC":
                until_pvc -= 1
                if until_pvc == 0:
                    rr, kind = 0.60 * rr, "V"
                    until_pvc = int(rng.integers(3, 6))
                    owe_pause = True
                elif owe_pause:
                    rr *= 1.40
                    owe_pause = False
        t += rr
        times.append(t)
        types.append(kind)
        rr_before.append(rr)
    return np.array(times), types, np.array(rr_before)


def synthesize_ecg(
    bpm: float = 72.0,
    duration_s: float = 10.0,
    fs: float = 250.0,
    snr_db: float = 30.0,
    wander_mv: float = 0.15,
    rhythm: str = "NSR",
    seed: int | None = None,
) -> EcgRecord:
    """
    Build a synthetic single-lead (Lead II-like) ECG.

    bpm         target mean heart rate, 40-180
    duration_s  window length; the PRD caps windows at 10-30 s so the UI stays responsive
    fs          sampling rate in Hz
    snr_db      additive white Gaussian noise (AWGN) level; np.inf disables noise
    wander_mv   amplitude of 0.5 Hz baseline wander (breathing moves the chest and the electrodes)
    rhythm      "NSR", "AFib" or "PVC"
    seed        makes the "random" signal reproducible
    """
    if rhythm not in RHYTHMS:
        raise ValueError(f"rhythm must be one of {RHYTHMS}")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs                     # sample n happens at time n / Fs

    beat_t, beat_kind, rr_before = _beat_schedule(rhythm, bpm, duration_s, rng)

    # ---- 1. Superpose the Gaussian waves of every beat --------------------------------------
    clean = np.zeros(n)
    for tk, kind, rr in zip(beat_t, beat_kind, rr_before):
        waves = dict(_PVC_WAVES if kind == "V" else _NORMAL_WAVES)
        if rhythm == "AFib":
            waves.pop("P", None)              # no organised atrial contraction -> no P wave
        # Faster heart rates shorten the P-R and Q-T spans.  A square-root rule (in the spirit of
        # Bazett's QT correction) scales the P and T offsets with the preceding RR interval.
        scale = float(np.clip(np.sqrt(rr / 0.83), 0.6, 1.4))
        # Only evaluate each beat where it is non-negligible (+/- 0.6 s) -- much faster than
        # evaluating every Gaussian over the whole window.
        lo, hi = max(0, int((tk - 0.6) * fs)), min(n, int((tk + 0.6) * fs) + 1)
        if hi <= lo:
            continue
        seg = t[lo:hi]
        for name, (amp, mu, sigma) in waves.items():
            offset = mu * scale if name in ("P", "T") else mu
            clean[lo:hi] += _gaussian(seg, amp, tk + offset, sigma)

    # AFib also shows fibrillatory "f-waves": a small, chaotic 4-8 Hz ripple on the baseline.
    if rhythm == "AFib":
        f_freq = rng.uniform(4.5, 7.5)
        envelope = 1.0 + 0.5 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
        clean += 0.04 * envelope * np.sin(2 * np.pi * f_freq * t + rng.uniform(0, 2 * np.pi))

    # ---- 2. Add noise -----------------------------------------------------------------------
    # SNR_dB = 10 log10(P_signal / P_noise)  =>  P_noise = P_signal / 10^(SNR_dB / 10)
    # For white noise, power equals variance, so sigma_noise = sqrt(P_noise).
    signal = clean.copy()
    if np.isfinite(snr_db):
        p_signal = float(np.mean(clean**2))
        sigma_noise = np.sqrt(p_signal / 10 ** (snr_db / 10.0))
        signal += rng.normal(0.0, sigma_noise, n)

    # Baseline wander: a 0.5 Hz sinusoid (30 breaths/min-ish) far below the ECG band, which is
    # exactly what the 0.5 Hz high-pass edge of the DSP filter is designed to remove.
    signal += wander_mv * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi))

    # ---- 3. Ground truth --------------------------------------------------------------------
    keep = beat_t < duration_s
    beat_t, beat_kind = beat_t[keep], [k for k, m in zip(beat_kind, keep) if m]

    if rhythm == "AFib":
        label = "Atrial Fibrillation"
    elif rhythm == "PVC":
        label = "PVC"
    else:
        label = rate_label(bpm)

    return EcgRecord(
        t=t, signal=signal, fs=float(fs), beat_times=beat_t, beat_types=beat_kind, label=label,
        source="synthetic",
        description=f"Synthetic {label}, {bpm:.0f} bpm target, SNR {snr_db:g} dB",
        params=dict(bpm=bpm, snr_db=snr_db, rhythm=rhythm, wander_mv=wander_mv, seed=seed),
    )


# ----------------------------------------------------------------------------------------------
# Mode B: MIT-BIH Arrhythmia Database (real patients) via PhysioNet
# ----------------------------------------------------------------------------------------------
# PRD note: it names a Hugging Face "mitbih" dataset, but those copies are mostly pre-cut single
# heartbeats.  We want *continuous* traces, so we read the original database from PhysioNet
# (https://physionet.org/content/mitdb/) with the `wfdb` package.
#
# MIT-BIH: 48 half-hour two-lead recordings at 360 Hz.  We use channel 0 (usually "MLII", a
# Lead II-like lead), and the cardiologists' beat annotations serve as ground truth.
MITBIH_RECORDS: dict[str, dict] = {
    "100": {"desc": "Mostly normal sinus rhythm", "start_s": 0},
    "119": {"desc": "Frequent PVCs (bigeminy)", "start_s": 0},
    "201": {"desc": "Atrial fibrillation with PVCs", "start_s": 0},
    "208": {"desc": "Frequent PVCs", "start_s": 0},
}

# Annotation symbols that denote an actual heartbeat (the rest are rhythm/noise markers).
_BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _infer_label(symbols: list[str], aux: list[str], bpm: float) -> str:
    """Summarise a real slice with one of our classes, using the annotations."""
    if any(a.strip("\x00").startswith("(AFIB") for a in aux):
        return "Atrial Fibrillation"
    if symbols and symbols.count("V") / len(symbols) >= 0.08:
        return "PVC"
    return rate_label(bpm)


def load_mitbih(record: str = "100", start_s: float | None = None, duration_s: float = 10.0) -> EcgRecord:
    """
    Fetch one slice of a MIT-BIH record.  The first call needs internet; the slice is then cached
    as a small .npz file in data/ so later calls are instant and work offline.
    """
    import wfdb  # imported lazily: only Mode B needs it (and the network)

    start_s = float(MITBIH_RECORDS.get(record, {}).get("start_s", 0) if start_s is None else start_s)
    cache = _DATA_DIR / f"mitdb_{record}_{start_s:g}_{duration_s:g}.npz"

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        signal, fs = z["signal"], float(z["fs"])
        beat_samples, symbols, aux = z["beat_samples"], list(z["symbols"]), list(z["aux"])
    else:
        info = wfdb.rdheader(record, pn_dir="mitdb")
        fs = float(info.fs)
        lo, hi = int(start_s * fs), int((start_s + duration_s) * fs)
        rec = wfdb.rdrecord(record, pn_dir="mitdb", sampfrom=lo, sampto=hi, channels=[0])
        ann = wfdb.rdann(record, "atr", pn_dir="mitdb", sampfrom=lo, sampto=hi)
        signal = rec.p_signal[:, 0].astype(float)
        # wfdb returns annotation sample numbers relative to the start of the file; shift them
        # so they index into our slice.
        samples = np.asarray(ann.sample)
        if samples.size and samples.min() >= lo:
            samples = samples - lo
        keep = [i for i, s in enumerate(ann.symbol) if s in _BEAT_SYMBOLS]
        beat_samples = samples[keep]
        symbols = [ann.symbol[i] for i in keep]
        aux = [(a or "") for a in ann.aux_note]
        _DATA_DIR.mkdir(exist_ok=True)
        np.savez_compressed(cache, signal=signal, fs=fs, beat_samples=beat_samples,
                            symbols=np.array(symbols), aux=np.array(aux))

    t = np.arange(len(signal)) / fs
    bpm = 60.0 / np.diff(beat_samples / fs).mean() if len(beat_samples) > 1 else 0.0
    return EcgRecord(
        t=t, signal=signal, fs=fs, beat_times=beat_samples / fs,
        beat_types=["V" if s == "V" else "N" for s in symbols],
        label=_infer_label(symbols, aux, bpm), source="mitbih",
        description=f"MIT-BIH record {record}: {MITBIH_RECORDS.get(record, {}).get('desc', '')}",
        params=dict(record=record, start_s=start_s, duration_s=duration_s),
    )
