"""
make_figures.py -- draws the illustrations used in docs/STUDENT_GUIDE.md.

Every curve is REAL output of this project's own code (signal_gen.py and dsp.py), so the pictures
match what students see in the app.  Plain SVG is written by hand (no plotting library needed).

Run from the repo root:   uv run python docs/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsp import analyze, bandpass  # noqa: E402
from signal_gen import _NORMAL_WAVES, _gaussian, synthesize_ecg  # noqa: E402

OUT = Path(__file__).resolve().parent / "img"
INK, MUTED, GRID = "#161616", "#6f6f6f", "#e0e0e0"
BLUE, RED, GREEN, ORANGE, PURPLE = "#0f62fe", "#da1e28", "#198038", "#ba4e00", "#8a3ffc"
FONT = "IBM Plex Sans, Segoe UI, Arial, sans-serif"


class Panel:
    """A rectangle on the canvas that maps data coordinates to pixels."""

    def __init__(self, svg, x, y, w, h, xr, yr, title="", xlabel="", ylabel=""):
        self.svg, self.x, self.y, self.w, self.h, self.xr, self.yr = svg, x, y, w, h, xr, yr
        svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff" stroke="{GRID}"/>')
        if title:
            svg.text(x, y - 8, title, size=13, weight=600)
        if xlabel:
            svg.text(x + w / 2, y + h + 26, xlabel, size=11, anchor="middle", fill=MUTED)
        if ylabel:
            svg.add(f'<text transform="translate({x - 38},{y + h / 2}) rotate(-90)" font-size="11" '
                    f'text-anchor="middle" fill="{MUTED}" font-family="{FONT}">{ylabel}</text>')

    def px(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def xticks(self, ticks):
        for t in ticks:
            self.svg.add(f'<line x1="{self.px(t):.1f}" y1="{self.y}" x2="{self.px(t):.1f}" y2="{self.y + self.h}" '
                         f'stroke="{GRID}" stroke-dasharray="2 3"/>')
            self.svg.text(self.px(t), self.y + self.h + 14, f"{t:g}", size=10, anchor="middle", fill=MUTED)

    def line(self, xs, ys, color=BLUE, width=1.6, max_pts=2500):
        step = max(1, len(xs) // max_pts)
        pts = " ".join(f"{self.px(a):.1f},{self.py(b):.1f}" for a, b in zip(xs[::step], ys[::step]))
        self.svg.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')

    def dots(self, xs, ys, color=RED, r=3.2, marker="circle"):
        for a, b in zip(xs, ys):
            cx, cy = self.px(a), self.py(b)
            if marker == "tri":
                self.svg.add(f'<polygon points="{cx - 5:.1f},{cy - 9:.1f} {cx + 5:.1f},{cy - 9:.1f} {cx:.1f},{cy - 1:.1f}" fill="{color}"/>')
            else:
                self.svg.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}"/>')


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, s):
        self.items.append(s)

    def text(self, x, y, s, size=12, anchor="start", fill=INK, weight=400):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
                 f'font-weight="{weight}" font-family="{FONT}">{s}</text>')

    def save(self, name):
        OUT.mkdir(exist_ok=True)
        body = "\n".join(self.items)
        (OUT / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n'
            f'<rect width="100%" height="100%" fill="#ffffff"/>\n{body}\n</svg>\n', encoding="utf-8")
        print("wrote", OUT / name)


def arrow_span(svg, p, t0, t1, ypix, label, color):
    """Horizontal double-headed span with a label, in data-x coordinates."""
    x0, x1 = p.px(t0), p.px(t1)
    svg.add(f'<line x1="{x0:.1f}" y1="{ypix}" x2="{x1:.1f}" y2="{ypix}" stroke="{color}" stroke-width="2"/>')
    for x in (x0, x1):
        svg.add(f'<line x1="{x:.1f}" y1="{ypix - 5}" x2="{x:.1f}" y2="{ypix + 5}" stroke="{color}" stroke-width="2"/>')
    svg.text((x0 + x1) / 2, ypix - 8, label, size=11, anchor="middle", fill=color, weight=600)


# ---------------------------------------------------------------------------------------------
def fig_beat():
    """One heartbeat, built from the five Gaussians, with the waves and intervals labelled."""
    t = np.linspace(-0.4, 0.62, 1500)
    y = sum(_gaussian(t, a, mu, s) for a, mu, s in _NORMAL_WAVES.values())
    svg = Svg(900, 420)
    svg.text(24, 30, "One heartbeat = five Gaussian bumps (P, Q, R, S, T)", size=16, weight=600)
    p = Panel(svg, 70, 60, 800, 300, (-0.4, 0.62), (-0.9, 1.25), xlabel="time around the beat (seconds)", ylabel="millivolts (mV)")
    p.xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    p.line(t, y, BLUE, 2.6)
    marks = {"P": (-0.16, 0.15, "P  atria fire", 0, -16), "Q": (-0.028, -0.12, "Q", -12, 4),
             "R": (0.0, 1.0, "R  the big spike", 70, 6), "S": (0.03, -0.22, "S", 12, 6),
             "T": (0.24, 0.30, "T  ventricles reset", 0, -18)}
    for k, (tx, ty, label, dx, dy) in marks.items():
        p.dots([tx], [ty], RED, 4)
        svg.text(p.px(tx) + dx, p.py(ty) + dy, label, size=12, anchor="middle", fill=RED, weight=600)
    base = p.py(-0.62)
    arrow_span(svg, p, -0.205, -0.05, base, "PR interval", GREEN)
    arrow_span(svg, p, -0.05, 0.055, base + 30, "QRS width", ORANGE)
    arrow_span(svg, p, -0.05, 0.33, base - 30, "QT interval", PURPLE)
    svg.text(24, 404, "Made with signal_gen.py (noise-free). Interval markers are approximate for this model beat.", size=11, fill=MUTED)
    svg.save("fig_beat.svg")


def fig_sampling():
    """Same heartbeat sampled too slowly and fast enough."""
    rec = synthesize_ecg(bpm=60, duration_s=3, fs=1000, snr_db=np.inf, wander_mv=0, seed=1)
    lo, hi = 1.0, 2.0
    m = (rec.t >= lo) & (rec.t <= hi)
    svg = Svg(900, 560)
    svg.text(24, 30, "Sampling: how often you look decides what you can see", size=16, weight=600)
    for row, (fs, color, note) in enumerate([(12, RED, "12 samples/second: the sharp R spike falls between the dots and is lost"),
                                             (250, GREEN, "250 samples/second: dots so close they trace the whole shape")]):
        p = Panel(svg, 70, 70 + row * 240, 800, 185, (lo, hi), (-0.45, 1.25), title=note, xlabel="seconds" if row else "",
                  ylabel="mV")
        p.xticks(np.arange(1.0, 1.01 + 1, 0.2))
        p.line(rec.t[m], rec.signal[m], "#c6c6c6", 3.0)              # the "true" continuous heartbeat
        idx = np.arange(int(lo * 1000), int(hi * 1000), int(round(1000 / fs)))
        p.line(rec.t[idx], rec.signal[idx], color, 1.4)
        p.dots(rec.t[idx], rec.signal[idx], color, 2.6 if fs > 100 else 4.2)
    svg.text(24, 552, "Grey = the real signal. Coloured dots/lines = what the computer actually gets (samples).", size=11, fill=MUTED)
    svg.save("fig_sampling.svg")


def fig_filtering():
    rec = synthesize_ecg(bpm=72, duration_s=10, fs=250, snr_db=14, wander_mv=0.35, seed=4)
    clean = bandpass(rec.signal, rec.fs, 0.5, 45)
    svg = Svg(900, 520)
    svg.text(24, 30, "Cleaning the signal: a 0.5 - 45 Hz band-pass filter", size=16, weight=600)
    a = Panel(svg, 70, 66, 800, 180, (0, 10), (-0.9, 1.6), title="Raw: slow drift (breathing) + fuzz (muscle noise)", ylabel="mV")
    a.xticks(range(0, 11, 2))
    a.line(rec.t, rec.signal, RED, 1.2)
    b = Panel(svg, 70, 290, 800, 180, (0, 10), (-0.9, 1.6), title="Filtered: baseline steady, fuzz gone, heartbeat timing unchanged",
              xlabel="seconds", ylabel="mV")
    b.xticks(range(0, 11, 2))
    b.line(rec.t, clean, BLUE, 1.5)
    svg.text(24, 510, "Zero-phase Butterworth filter from dsp.py (filtered forwards then backwards, so peaks do not shift in time).", size=11, fill=MUTED)
    svg.save("fig_filtering.svg")


def fig_pantompkins():
    rec = synthesize_ecg(bpm=78, duration_s=6, fs=250, snr_db=20, wander_mv=0.2, seed=6)
    a = analyze(rec.signal, rec.fs)
    stages = [("1  band-pass 5-15 Hz: keep only the QRS 'sound'", a.pt.band),
              ("2  derivative: how steep is the trace?", a.pt.derivative),
              ("3  squared: make big slopes HUGE, small ones vanish", a.pt.squared),
              ("4  moving-window integral: one smooth hill per heartbeat", a.pt.integrated)]
    svg = Svg(900, 785)
    svg.text(24, 30, "Pan-Tompkins: turn 'find the heartbeat' into 'find the tall hills'", size=16, weight=600)
    for i, (title, y) in enumerate(stages):
        p = Panel(svg, 70, 66 + i * 168, 800, 128, (0, 6), (float(y.min()) - 0.1 * abs(y).max(), float(y.max()) * 1.18),
                  title=title, xlabel="seconds" if i == 3 else "")
        p.xticks(range(0, 7))
        p.line(rec.t, y, BLUE, 1.4)
        p.dots(rec.t[a.peaks], y[a.peaks], GREEN, marker="tri")
    svg.text(24, 772, "Green triangles = detected R peaks. Real output of dsp.pan_tompkins() on a synthetic 78 bpm ECG.", size=11, fill=MUTED)
    svg.save("fig_pantompkins.svg")


def fig_nsr_vs_afib():
    svg = Svg(900, 520)
    svg.text(24, 30, "Normal rhythm vs atrial fibrillation (same 75 bpm average)", size=16, weight=600)
    for row, (rhythm, title) in enumerate([("NSR", "Normal sinus rhythm: even spacing, a small P bump before every spike"),
                                           ("AFib", "Atrial fibrillation: uneven spacing, no P bumps, fuzzy baseline")]):
        rec = synthesize_ecg(bpm=75, duration_s=10, fs=250, snr_db=30, wander_mv=0.05, rhythm=rhythm, seed=2)
        a = analyze(rec.signal, rec.fs)      # analyse 10 s but draw only the first 8: filters misbehave at the edges
        vis = rec.t <= 8.0
        pk = a.peaks[rec.t[a.peaks] <= 8.0]
        p = Panel(svg, 70, 66 + row * 220, 800, 170, (0, 8), (-0.5, 1.45), title=title,
                  xlabel="seconds" if row else "", ylabel="mV")
        p.xticks(range(0, 9))
        p.line(rec.t[vis], a.filtered[vis], BLUE, 1.5)
        p.dots(rec.t[pk], a.filtered[pk], GREEN, marker="tri")
        for k in range(min(7, len(a.peaks) - 1)):            # label the R-R gaps in milliseconds
            x0, x1 = rec.t[a.peaks[k]], rec.t[a.peaks[k + 1]]
            if x1 > 8:
                break
            svg.text(p.px((x0 + x1) / 2), p.py(-0.36), f"{(x1 - x0) * 1000:.0f} ms", size=10, anchor="middle",
                     fill=ORANGE, weight=600)
    svg.text(24, 510, "Orange numbers are the R-R intervals (time between beats). Steady numbers = regular. Jumpy numbers = irregular.", size=11, fill=MUTED)
    svg.save("fig_nsr_vs_afib.svg")


if __name__ == "__main__":
    for f in (fig_beat, fig_sampling, fig_filtering, fig_pantompkins, fig_nsr_vs_afib):
        f()
