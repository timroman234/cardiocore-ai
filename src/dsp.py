"""
dsp.py -- Digital signal processing for the EKG: clean it, find the heartbeats, measure them.

Pipeline (PRD section 4):

    raw ECG --bandpass 0.5-45 Hz--> clean ECG --Pan-Tompkins--> R-peak indices
                                                     |
                     R-R intervals -> HR, SDNN, RMSSD, pNN50, ...     (heart-rate variability)
                     waveform windows -> QRS width, PR, QT, P-wave     (delineation estimates)
                     FFT / Welch -> power spectral density

Sampling-rate refresher: with sampling rate Fs, sample index n happens at time t = n / Fs, so a
duration in seconds converts to samples as  n = round(seconds * Fs).  The highest frequency a
sampled signal can hold is the Nyquist frequency Fs / 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps


# ----------------------------------------------------------------------------------------------
# 1. Filtering
# ----------------------------------------------------------------------------------------------
def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter.

    Why 0.5 - 45 Hz for a diagnostic-style ECG?
      * Below ~0.5 Hz lives *baseline wander*: breathing, sweat, and electrode movement slowly
        push the whole trace up and down.  It is not cardiac, and it can swamp the ST/T-wave.
      * Above ~45 Hz lives *muscle tremor* (EMG) and mains hum (50/60 Hz).  Almost no useful QRS
        energy is up there, so cutting it costs little and removes a lot of noise.

    Why Butterworth?  It has the flattest possible pass-band (no ripple), so wave amplitudes are not
    distorted.  `order` sets how steeply it rolls off past the cut-offs; the filter is designed as
    cascaded second-order sections (SOS), which stays numerically stable for narrow bands.

    Why *zero-phase* (`sosfiltfilt`)?  A normal causal filter delays different frequencies by
    different amounts, smearing wave *timing*.  Running the filter forward and then backward
    cancels the delay, so R peaks stay exactly where they were.  That matters when we measure
    milliseconds.  (The price: it is not causal, so it can't be used on a live stream.)

    Cut-offs must satisfy 0 < low < high < Nyquist = Fs / 2.
    """
    nyquist = fs / 2.0
    if not (0 < low < high < nyquist):
        raise ValueError(f"need 0 < low < high < Nyquist ({nyquist} Hz); got {low}, {high}")
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass, used to smooth the signal before measuring wave boundaries."""
    sos = sps.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


# ----------------------------------------------------------------------------------------------
# 2. Pan-Tompkins QRS detection
# ----------------------------------------------------------------------------------------------
@dataclass
class PanTompkinsResult:
    peaks: np.ndarray        # R-peak sample indices
    band: np.ndarray         # stage 0: 5-15 Hz band-passed ECG
    derivative: np.ndarray   # stage 1
    squared: np.ndarray      # stage 2
    integrated: np.ndarray   # stage 3: moving-window integral (what the thresholds run on)
    threshold: np.ndarray    # the adaptive threshold value each detected beat was compared with


def pan_tompkins(x: np.ndarray, fs: float) -> PanTompkinsResult:
    """
    Pan & Tompkins (1985) real-time QRS detector, simplified.  Idea: turn "find the QRS" into
    "find the big bumps" by making the QRS the only loud thing left in the signal.

      0. Band-pass 5-15 Hz.  QRS energy concentrates here; P/T waves (slow) and noise (fast)
         are attenuated.
      1. Derivative.  The QRS has the steepest slopes in the ECG, so |dx/dt| is large there.
         We use the classic 5-point derivative:
             y[n] = Fs/8 * ( -x[n-2] - 2 x[n-1] + 2 x[n+1] + x[n+2] )
      2. Squaring.  y^2 is always positive and exaggerates large slopes (QRS) relative to small
         ones (T wave, noise).
      3. Moving-window integration.  Averaging over ~150 ms (about one QRS width) merges each
         QRS's slopes into a single smooth hill whose peak marks the beat:
             z[n] = (1/W) * sum_{k=0}^{W-1} y^2[n-k]
         (centred here so the hill sits on the QRS instead of lagging behind it).
      4. Adaptive thresholding.  Track a running estimate of the typical *signal* peak height
         (SPKI) and *noise* peak height (NPKI), and call a hill a beat if it is above
             THR = NPKI + 0.25 * (SPKI - NPKI)
         Both estimates update with an exponential average, so the detector adapts when the
         amplitude or noise level changes.  A 200 ms refractory period stops one QRS being
         counted twice (the heart physically cannot re-fire that fast).
    """
    band = bandpass(x, fs, 5.0, 15.0, order=2)

    # 5-point derivative (a convolution with the kernel below), scaled by Fs/8 to get units/second.
    kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * fs / 8.0
    derivative = np.convolve(band, kernel[::-1], mode="same")

    squared = derivative**2

    win = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    # Candidate hills: local maxima at least 200 ms apart (the refractory period).
    cand, _ = sps.find_peaks(integrated, distance=max(1, int(0.200 * fs)))

    peaks_mwi: list[int] = []
    thresholds: list[float] = []
    if len(cand):
        # Initialise the running estimates from the first two seconds of the hill signal.
        seg = integrated[: int(2 * fs)]
        spki, npki = float(seg.max()), float(seg.mean())
        thr = npki + 0.25 * (spki - npki)
        rr_avg = 0.8 * fs  # samples; refined as beats accumulate
        for c in cand:
            v = float(integrated[c])
            if v > thr:
                peaks_mwi.append(int(c))
                thresholds.append(thr)
                spki = 0.125 * v + 0.875 * spki
                if len(peaks_mwi) >= 2:
                    rr_avg = 0.75 * rr_avg + 0.25 * (peaks_mwi[-1] - peaks_mwi[-2])
            else:
                npki = 0.125 * v + 0.875 * npki
            thr = npki + 0.25 * (spki - npki)

        # Search-back: if a long gap opened up (> 166% of the average RR) we probably raised the
        # bar too high and skipped a real beat.  Re-scan the gap with half the threshold.
        peaks_mwi = _search_back(peaks_mwi, cand, integrated, thresholds, rr_avg)

    # Refine: each hill peak only says "a QRS is near here".  The R peak itself is the largest
    # |deflection| of the cleaned ECG within +/-80 ms of the hill peak.
    half = int(0.080 * fs)
    xa = np.abs(x)
    peaks = []
    for p in peaks_mwi:
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        peaks.append(lo + int(np.argmax(xa[lo:hi])))
    peaks = np.unique(np.array(peaks, dtype=int))

    return PanTompkinsResult(peaks, band, derivative, squared, integrated, np.array(thresholds))


def _search_back(peaks, cand, integrated, thresholds, rr_avg):
    """Re-inspect abnormally long gaps between detected beats using a lower (half) threshold."""
    if len(peaks) < 2:
        return peaks
    thr_half = 0.5 * np.median(thresholds) if len(thresholds) else 0.0
    out = [peaks[0]]
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b - a > 1.66 * rr_avg:
            inside = [c for c in cand if a + 0.2 * rr_avg < c < b - 0.2 * rr_avg]
            if inside:
                best = max(inside, key=lambda c: integrated[c])
                if integrated[best] > thr_half:
                    out.append(int(best))
        out.append(b)
    return out


def score_detection(detected_s: np.ndarray, truth_s: np.ndarray, tol_ms: float = 50.0,
                    margin_s: float = 0.3, duration_s: float | None = None) -> dict:
    """
    Compare detected beat times with ground truth (one-to-one matching within +/- tol_ms).

        sensitivity = TP / (TP + FN)   "of the real beats, how many did we find?"
        PPV         = TP / (TP + FP)   "of the beats we reported, how many were real?"

    Beats within `margin_s` of the window edges are ignored (filters misbehave at the edges).
    """
    tol = tol_ms / 1000.0
    truth = np.asarray(truth_s, float)
    det = np.asarray(detected_s, float)
    end = duration_s if duration_s is not None else max(truth.max(initial=0), det.max(initial=0))
    truth = truth[(truth >= margin_s) & (truth <= end - margin_s)]
    det = det[(det >= margin_s) & (det <= end - margin_s)]
    used = np.zeros(len(det), bool)
    tp = 0
    for t in truth:
        if len(det) == 0:
            break
        j = int(np.argmin(np.abs(det - t) + used * 1e9))
        if abs(det[j] - t) <= tol and not used[j]:
            used[j] = True
            tp += 1
    fn, fp = len(truth) - tp, len(det) - tp
    return dict(tp=tp, fp=fp, fn=fn,
                sensitivity=tp / len(truth) if len(truth) else float("nan"),
                ppv=tp / len(det) if len(det) else float("nan"))


# ----------------------------------------------------------------------------------------------
# 3. Heart-rate variability from R-R intervals
# ----------------------------------------------------------------------------------------------
def rr_features(peaks: np.ndarray, fs: float) -> dict[str, float]:
    """
    The R-R interval (RR) is the time between consecutive R peaks.  Heart rate is its inverse:
        HR [bpm] = 60 / RR [s]
    Heart-rate variability (HRV) summarises how much RR changes from beat to beat:
        SDNN   = std(RR)                                   overall variability
        RMSSD  = sqrt(mean( (RR[i+1] - RR[i])^2 ))         beat-to-beat (short-term) variability
        pNN50  = % of successive differences > 50 ms       another short-term measure
    Regular rhythms give small values, atrial fibrillation gives large ones.
    """
    rr = np.diff(peaks) / fs * 1000.0  # ms
    if len(rr) < 2:
        return dict(valid=False, heart_rate_bpm=0.0, mean_rr_ms=0.0, sdnn_ms=0.0, rmssd_ms=0.0,
                    pnn50=0.0, rr_cv=0.0, rr_min_max_ratio=1.0, rmssd_norm=0.0)
    d = np.diff(rr)
    mean_rr = float(rr.mean())
    return dict(
        valid=True,
        heart_rate_bpm=60000.0 / mean_rr,
        mean_rr_ms=mean_rr,
        sdnn_ms=float(rr.std(ddof=1)),
        rmssd_ms=float(np.sqrt(np.mean(d**2))),
        pnn50=float(100.0 * np.mean(np.abs(d) > 50.0)),
        rr_cv=float(rr.std(ddof=1) / mean_rr),              # coefficient of variation = SDNN / mean
        rr_min_max_ratio=float(rr.min() / rr.max()),
        rmssd_norm=float(np.sqrt(np.mean(d**2)) / mean_rr),  # RMSSD relative to the mean RR
    )


def spectrum(x: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density (PSD) via Welch's method: chop the signal into overlapping segments,
    take the Fourier transform of each,
        X[k] = sum_n x[n] * exp(-j 2 pi k n / N)        (the DFT, computed with the FFT algorithm)
    and average |X[k]|^2 across segments.  The result shows *how much power sits at each
    frequency* -- for an ECG: a peak at the heart rate (~1 Hz), then harmonics from the QRS.
    """
    nperseg = nperseg or int(min(len(x), 4 * fs))
    return sps.welch(x, fs=fs, nperseg=nperseg)


def lf_hf_ratio(peaks: np.ndarray, fs: float) -> float:
    """
    Classic HRV frequency-domain measure: power in the low band (0.04-0.15 Hz, sympathetic +
    parasympathetic) divided by the high band (0.15-0.4 Hz, breathing / parasympathetic).
    Needs a long recording to resolve 0.04 Hz, so we return NaN for windows under 20 s.
    """
    if len(peaks) < 8 or (peaks[-1] - peaks[0]) / fs < 20:
        return float("nan")
    t = peaks[1:] / fs
    rr = np.diff(peaks) / fs
    grid = np.arange(t[0], t[-1], 0.25)                      # resample the tachogram at 4 Hz
    rr_u = np.interp(grid, t, rr)
    f, p = sps.welch(rr_u - rr_u.mean(), fs=4.0, nperseg=min(len(rr_u), 64))
    lf = np.trapezoid(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapezoid(p[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)])
    return float(lf / hf) if hf > 0 else float("nan")


# ----------------------------------------------------------------------------------------------
# 4. Waveform delineation: QRS width, PR, QT, P-wave presence  (ESTIMATES)
# ----------------------------------------------------------------------------------------------
# Real clinical software delineates waves with far more elaborate methods (and 12 leads).  These
# are simple, readable heuristics built for teaching; treat the outputs as estimates.
_QRS_SLOPE_FRAC = 0.25   # QRS spans where |slope| exceeds this fraction of its peak slope (tuned on the synthetic model)
_WIDE_QRS_MS = 120.0     # >= 120 ms is "wide" by the usual clinical definition


def _edge(sl: np.ndarray, start: int, step: int, thr: float, limit: int, quiet_n: int) -> int:
    """
    Find where the QRS ends, walking from the R peak in direction `step` (+1 forward, -1 back).
    The slope is ~0 right at the R peak (it is a maximum!), so first walk into the steep flank,
    then keep going until the slope has stayed below `thr` for `quiet_n` samples (long enough to
    step over the brief slope-zero at the tip of the S or Q wave).
    """
    i = start
    while 0 < i < len(sl) - 1 and abs(i - start) < limit and sl[i] < thr:
        i += step                      # phase 1: climb onto the steep flank
    quiet = 0
    while 0 < i < len(sl) - 1 and abs(i - start) < limit:
        quiet = quiet + 1 if sl[i] < thr else 0
        if quiet >= quiet_n:
            return i - step * (quiet_n - 1)   # first quiet sample = boundary
        i += step
    return i


def delineate(x_raw: np.ndarray, peaks: np.ndarray, fs: float) -> dict[str, float]:
    """
    Estimate QRS duration, PR interval, QT interval and P-wave presence.

    Works on a copy of the ECG smoothed to <= 20 Hz (so noise doesn't create false boundaries).

      QRS width : the QRS is where the trace is *steep*.  Walk outward from the R peak until the
                  slope |dx/dt| drops below 25% of its peak value on each side.
                  A *wide* QRS (>= 120 ms) suggests the beat began in the ventricle (PVC).
      P wave    : search the gap before the QRS for a bump that is clearly above the noise floor.
                  Missing in atrial fibrillation (no coordinated atrial contraction).
      PR        : from P-wave onset to QRS onset (atrial -> ventricular conduction time).
      QT        : from QRS onset to the end of the T wave (ventricular depolarisation +
                  repolarisation).  Depends on heart rate, so clinicians "correct" it (QTc).

    Per-beat values are combined with the *median* so a couple of odd beats (or a PVC) don't
    drag the summary.
    """
    out = dict(qrs_duration_ms=float("nan"), wide_qrs_fraction=0.0, pr_interval_ms=float("nan"),
               qt_interval_ms=float("nan"), p_wave_present=False, p_wave_amplitude_mv=0.0)
    if len(peaks) < 3:
        return out

    xs = lowpass(x_raw, fs, 20.0)
    slope = np.abs(np.gradient(xs)) * fs
    # noise level of the raw trace, robust to the sparse QRS spikes: median(|diff|) / 0.6745 / sqrt(2)
    noise_raw = float(np.median(np.abs(np.diff(x_raw))) / 0.6745 / np.sqrt(2))
    noise_smooth = noise_raw * np.sqrt(20.0 / (fs / 2.0))   # white noise shrinks with bandwidth
    p_thr = max(0.09, 4.0 * noise_smooth)                   # mV; a P must beat this to count

    widths, p_amps, pr_list, qt_list = [], [], [], []
    for k in range(1, len(peaks) - 1):
        r, r_prev, r_next = int(peaks[k]), int(peaks[k - 1]), int(peaks[k + 1])
        rr_prev, rr_next = (r - r_prev) / fs, (r_next - r) / fs

        # --- QRS onset / offset from the slope profile ---------------------------------------
        lo, hi = max(0, r - int(0.15 * fs)), min(len(xs), r + int(0.15 * fs))
        thr = _QRS_SLOPE_FRAC * slope[lo:hi].max()
        limit, quiet_n = int(0.15 * fs), max(2, int(0.02 * fs))
        on = _edge(slope, r, -1, thr, limit, quiet_n)
        off = _edge(slope, r, +1, thr, limit, quiet_n)
        widths.append((off - on) / fs * 1000.0)

        # --- P wave: window between the previous T wave and the QRS onset --------------------
        w_lo = r - int(min(0.32, 0.45 * rr_prev) * fs)
        w_hi = on - int(0.02 * fs)
        if w_hi - w_lo >= int(0.04 * fs):
            w = xs[w_lo:w_hi]
            m = max(2, len(w) // 8)
            base = np.linspace(w[:m].mean(), w[-m:].mean(), len(w))  # straight-line baseline
            res = w - base
            ip = int(np.argmax(res))
            p_amps.append(float(res[ip]))
            # P onset: last point before the peak where the bump is < 20% of its height
            below = np.where(res[:ip] < 0.2 * res[ip])[0]
            p_on = w_lo + (int(below[-1]) if len(below) else 0)
            pr_list.append((on - p_on) / fs * 1000.0)

        # --- T wave end -> QT ----------------------------------------------------------------
        t_lo, t_hi = off + int(0.03 * fs), r + int(min(0.55, 0.65 * rr_next) * fs)
        if t_hi - t_lo >= int(0.06 * fs):
            seg = xs[t_lo:t_hi]
            base = float(np.median(seg[-max(2, int(0.04 * fs)):]))
            res = np.abs(seg - base)
            it = int(np.argmax(res))
            after = np.where(res[it:] < 0.2 * res[it])[0]
            t_end = t_lo + it + (int(after[0]) if len(after) else len(res) - it - 1)
            qt_list.append((t_end - on) / fs * 1000.0)

    widths = np.array(widths)
    out["qrs_duration_ms"] = float(np.median(widths))
    out["wide_qrs_fraction"] = float(np.mean(widths >= _WIDE_QRS_MS))
    if p_amps:
        amp = float(np.median(p_amps))
        out["p_wave_amplitude_mv"] = amp
        out["p_wave_present"] = bool(amp > p_thr)
        if out["p_wave_present"] and pr_list:
            out["pr_interval_ms"] = float(np.median(pr_list))
    if qt_list:
        out["qt_interval_ms"] = float(np.median(qt_list))
    return out


# ----------------------------------------------------------------------------------------------
# 5. One call that does everything
# ----------------------------------------------------------------------------------------------
@dataclass
class Analysis:
    filtered: np.ndarray
    pt: PanTompkinsResult
    features: dict[str, float]
    spectrum_f: np.ndarray
    spectrum_p: np.ndarray

    @property
    def peaks(self) -> np.ndarray:
        return self.pt.peaks


def analyze(signal: np.ndarray, fs: float) -> Analysis:
    """Filter -> detect -> measure.  Returns everything the UI and the ML model need."""
    filtered = bandpass(signal, fs, 0.5, 45.0)
    pt = pan_tompkins(filtered, fs)
    feats = {**rr_features(pt.peaks, fs), **delineate(signal, pt.peaks, fs)}
    feats["lf_hf"] = lf_hf_ratio(pt.peaks, fs)
    f, p = spectrum(filtered, fs)
    return Analysis(filtered, pt, feats, f, p)
