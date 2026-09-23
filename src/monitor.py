"""
monitor.py -- "Live replay": an animated bedside-monitor view of the window on screen.

How it works (and what it is NOT)
---------------------------------
Python has already filtered the signal and found the R peaks (dsp.py).  This module only packages
those results as JSON and hands them to a small piece of JavaScript that draws them on an HTML
<canvas> at ~60 frames per second, the way a hospital monitor does:

  * a bright dot (the "sweep cursor") moves left to right across a 5-second screen,
  * the trace is drawn behind it, and a short gap ahead of it is blanked (the "eraser"),
  * when the cursor reaches a heartbeat, a marker appears and the heart-rate readout updates.

Because the animation runs in the browser, it is smooth and does not re-run Python on every frame.

IMPORTANT (honesty): this is a REPLAY of a finished window.  The beats were found offline, with a
zero-phase filter that looks at the whole recording, so the markers appear "on time".  A real
monitor only knows the past, so its filter delays the trace and its detector fires slightly late.
docs/STUDENT_GUIDE.md section 6 explains that difference.

Security note: the HTML is built only from numbers and one short label, and the JSON is escaped so
that no value can close the <script> tag.
"""
from __future__ import annotations

import json

import numpy as np

SCREEN_SECONDS = 5.0     # width of the monitor screen; the window is split into whole sweeps of this size


def monitor_payload(rec, a, label: str | None = None) -> dict:
    """
    Everything the browser needs to animate one analysed window.

    rec    the EcgRecord (sampling rate, source)
    a      the dsp.Analysis (filtered signal and R-peak indices)
    """
    sig = np.asarray(a.filtered, float)
    fs = float(rec.fs)
    duration = len(sig) / fs

    # Split the window into an integer number of sweeps of about 5 s, so that the animation loops
    # seamlessly: 10 s -> 2 sweeps of 5 s, 30 s -> 6 sweeps of 5 s.
    sweeps = max(1, round(duration / SCREEN_SECONDS))
    screen_s = duration / sweeps

    peaks_t = np.asarray(a.peaks, float) / fs

    # Rolling heart rate at each beat: 60 / (mean of the last up to 4 R-R intervals).  The first
    # beat has no interval yet, so its rate is unknown (None -> "--" on screen).
    bpm: list[int | None] = [None]
    rr = np.diff(peaks_t)
    for i in range(1, len(peaks_t)):
        recent = rr[max(0, i - 4):i]
        bpm.append(int(round(60.0 / recent.mean())) if len(recent) and recent.mean() > 0 else None)

    lo, hi = float(sig.min()), float(sig.max())
    pad = 0.12 * max(hi - lo, 1e-6)

    if label is None:
        label = ("simulated Lead II" if rec.source == "synthetic"
                 else f"MIT-BIH record {rec.params.get('record', '')}, MLII")

    return dict(
        fs=fs,
        duration_s=round(duration, 4),
        screen_s=round(screen_s, 4),
        signal=[round(float(v), 3) for v in sig],
        peaks_t=[round(float(t), 4) for t in peaks_t],
        bpm_at_peak=bpm,
        y_min=round(lo - pad, 3),
        y_max=round(hi + pad, 3),
        label=label,
    )


def monitor_html(payload: dict, height: int = 374) -> str:
    """A complete, self-contained HTML page (canvas + JavaScript) for st.iframe()."""
    data = json.dumps(payload, separators=(",", ":"))
    data = data.replace("</", "<\\/")     # "\/" is legal JSON for "/": stops any value closing the <script> tag
    return _TEMPLATE.replace("__DATA__", data).replace("__HEIGHT__", str(int(height)))


# The page.  Plain HTML/CSS/JavaScript (no libraries).  Colours follow the app's Carbon dark theme;
# the trace is green like a classic patient monitor.
_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; background: #161616; overflow: hidden;
               font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif; color: #c6c6c6; }
  #wrap { height: __HEIGHT__px; display: flex; flex-direction: column; }
  canvas { flex: 1; width: 100%; min-height: 0; display: block; }
  #bar { height: 38px; display: flex; align-items: center; gap: 8px; padding: 0 2px;
         box-sizing: border-box; font-size: 12px; }
  button { background: transparent; color: #4589ff; border: 1px solid #4589ff; border-radius: 0;
           padding: 3px 12px; font-size: 12px; cursor: pointer; font-family: inherit; }
  button:hover { background: #0f62fe; color: #fff; }
  #note { color: #8d8d8d; margin-left: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style></head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="bar">
    <button id="play">Pause</button>
    <button id="speed">Speed 1x</button>
    <button id="snd">Sound off</button>
    <span id="note">Replay of the analysed window: beats were found offline, a real monitor detects them slightly late.</span>
  </div>
</div>
<script>
const DATA = __DATA__;
(function () {
  'use strict';
  const D = DATA, fs = D.fs, sig = D.signal, N = sig.length, dur = D.duration_s, S = D.screen_s;
  const peaks = D.peaks_t, bpm = D.bpm_at_peak;
  const GREEN = '#42be65', BEAT = '#a7f0ba', GRID = '#242424', GRID_MAJOR = '#333333', TEXT = '#8d8d8d';
  const PAD = { l: 44, r: 14, t: 34, b: 22 };
  const GAP = Math.min(0.45, 0.09 * S);          // seconds of blank "eraser" ahead of the cursor

  const cv = document.getElementById('c'), ctx = cv.getContext('2d');
  let W = 0, H = 0;
  function resize() {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(resize).observe(cv); resize();

  // data -> pixels
  const yOf = v => PAD.t + (D.y_max - v) / (D.y_max - D.y_min) * (H - PAD.t - PAD.b);
  const xOf = ph => PAD.l + ph / S * (W - PAD.l - PAD.r);
  // Signal value at time ts (seconds). The window loops, so negative / long times wrap around.
  const idxAt = ts => (((Math.floor(ts * fs)) % N) + N) % N;
  const valAt = ts => sig[idxAt(ts)];

  // playback state
  let t = 0, last = null, running = true, speed = 1, beatIdx = 0, pulse = 0, hr = null;
  const SPEEDS = [1, 0.5, 2];
  let audio = null, sound = false;

  function beep() {
    if (!sound || !audio) return;
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = 880; g.gain.value = 0.05;
    o.connect(g); g.connect(audio.destination);
    o.start(); o.stop(audio.currentTime + 0.06);
  }

  function onBeat(i) { pulse = 1; if (bpm[i] !== null) hr = bpm[i]; beep(); }   // keep the last rate on loop restart

  // Draw the samples whose time lies in [t0, t1); ts - offset is the position on the screen (seconds).
  // The glow is a wide, faint copy of the line drawn underneath (much cheaper than canvas shadowBlur).
  function trace(t0, t1, offset) {
    const i0 = Math.ceil(t0 * fs), i1 = Math.floor(t1 * fs);
    if (i1 <= i0) return;
    const path = new Path2D();
    for (let i = i0; i <= i1; i++) {
      const ts = i / fs, x = xOf(ts - offset), y = yOf(valAt(ts));
      if (i === i0) path.moveTo(x, y); else path.lineTo(x, y);
    }
    ctx.globalAlpha = 0.18; ctx.lineWidth = 6; ctx.stroke(path);
    ctx.globalAlpha = 1;    ctx.lineWidth = 2; ctx.stroke(path);
  }

  function grid() {
    ctx.lineWidth = 1;
    for (let s = 0; s <= S + 1e-9; s += 0.2) {                    // 0.2 s = one "big box" of ECG paper
      const major = Math.abs(s - Math.round(s)) < 1e-6;
      ctx.strokeStyle = major ? GRID_MAJOR : GRID;
      const x = Math.round(xOf(s)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
      if (major) { ctx.fillStyle = TEXT; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
                   ctx.fillText(Math.round(s) + ' s', x, H - 6); }
    }
    ctx.textAlign = 'right'; ctx.font = '10px sans-serif';
    for (let v = Math.ceil(D.y_min * 2) / 2; v <= D.y_max; v += 0.5) {     // horizontal lines every 0.5 mV
      const y = Math.round(yOf(v)) + 0.5;
      ctx.strokeStyle = GRID_MAJOR;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
      ctx.fillStyle = TEXT; ctx.fillText(v.toFixed(1), PAD.l - 6, y + 3);
    }
  }

  function draw() {
    ctx.fillStyle = '#0f0f0f'; ctx.fillRect(0, 0, W, H);
    grid();

    const k = Math.floor(t / S), start = k * S, phase = t - start;
    ctx.lineJoin = 'round'; ctx.strokeStyle = GREEN;
    trace(start, t, start);                                        // this sweep, up to the cursor
    trace(start - S + phase + GAP, start, start - S);              // previous sweep, after the eraser gap

    // heartbeat markers for beats that are currently on screen (the window loops, so also check p - dur)
    ctx.fillStyle = BEAT;
    for (const p0 of peaks) {
      for (const p of [p0, p0 - dur]) {
        let ph = null;
        if (p >= start && p <= t) ph = p - start;
        else if (p >= start - S + phase + GAP && p < start) ph = p - (start - S);
        if (ph === null) continue;
        const x = xOf(ph), y = yOf(valAt(p)) - 10;
        ctx.beginPath(); ctx.moveTo(x - 4, y - 7); ctx.lineTo(x + 4, y - 7); ctx.lineTo(x, y); ctx.closePath(); ctx.fill();
      }
    }

    // the sweep cursor
    const cx = xOf(phase), cy = yOf(valAt(t));
    ctx.fillStyle = GREEN; ctx.globalAlpha = 0.3;
    ctx.beginPath(); ctx.arc(cx, cy, 8, 0, 6.2832); ctx.fill();    // soft halo
    ctx.globalAlpha = 1; ctx.fillStyle = '#ffffff';
    ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, 6.2832); ctx.fill();

    // readouts
    ctx.textAlign = 'left'; ctx.fillStyle = TEXT; ctx.font = '11px sans-serif';
    ctx.fillText('REPLAY  |  ' + D.label + '  |  t = ' + t.toFixed(1) + ' / ' + dur.toFixed(0) + ' s', PAD.l, 18);
    ctx.textAlign = 'right';
    ctx.fillStyle = GREEN; ctx.font = '600 24px sans-serif';
    ctx.fillText(hr === null ? '--' : String(hr), W - PAD.r - 34, 27);
    ctx.font = '11px sans-serif'; ctx.fillStyle = TEXT; ctx.fillText('bpm', W - PAD.r, 27);
    ctx.fillStyle = GREEN; ctx.globalAlpha = 0.35 + 0.65 * pulse;   // pulsing dot: flashes on every beat
    ctx.beginPath(); ctx.arc(W - PAD.r - 96, 19, 4 + 4 * pulse, 0, 6.2832); ctx.fill();
    ctx.globalAlpha = 1;
  }

  function frame(now) {
    if (last === null) last = now;
    const real = Math.min(0.25, (now - last) / 1000); last = now;  // clamp: no big jump after a hidden tab, but real time down to ~4 fps
    if (running) {
      t += real * speed;
      if (t >= dur) { t -= dur; beatIdx = 0; }
      while (beatIdx < peaks.length && peaks[beatIdx] <= t) { onBeat(beatIdx); beatIdx++; }
    }
    pulse = Math.max(0, pulse - real * 4);
    draw();
    requestAnimationFrame(frame);
  }

  document.getElementById('play').onclick = function () {
    running = !running; this.textContent = running ? 'Pause' : 'Play';
  };
  document.getElementById('speed').onclick = function () {
    speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; this.textContent = 'Speed ' + speed + 'x';
  };
  document.getElementById('snd').onclick = function () {
    sound = !sound;
    if (sound && !audio) audio = new (window.AudioContext || window.webkitAudioContext)();   // needs this click
    this.textContent = sound ? 'Sound on' : 'Sound off';
  };

  requestAnimationFrame(frame);
})();
</script></body></html>
"""
