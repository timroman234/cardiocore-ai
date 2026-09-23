# CardioCore AI

Interactive time-series EKG signal processing and agentic diagnostic tutor: a single-screen
Streamlit dashboard for learning DSP, feature-based ML and LLM agents on cardiac signals.

> **Educational simulation only. Not a medical device.** Never use it for real diagnosis or triage.

## Quickstart

```bash
uv sync
cp .env.example .env          # then put your ANTHROPIC_API_KEY in .env
uv run streamlit run src/app.py
```

The first launch trains the rhythm classifier on synthetic ECGs (about a minute, cached in `models/`).
The MIT-BIH mode downloads a small slice from PhysioNet on first use (cached in `data/`).
Without an API key everything works except the "Ask Claude tutor" button.

## What you can do

| Area | Try this |
|---|---|
| Synthetic ECG | Sweep heart rate 40-180 bpm, lower the SNR to 5 dB and watch peak detection degrade, inject AFib or PVCs |
| Live replay | Switch the view to *Live replay* for an animated bedside-monitor sweep with beat markers, a rolling bpm readout and optional beep |
| Pan-Tompkins | Switch the view to *Pan-Tompkins stages* to see band-pass, derivative, squaring and integration |
| Real ECG | Switch to *MIT-BIH (real)*: detection is scored against cardiologist annotations |
| Domain gap | The classifier is trained on synthetic data only; on real records it can be confidently wrong |
| AI tutor | *Ask Claude tutor* returns a structured lesson (summary, waves, mechanism, red flags, quiz) |

## Learn with it

**New to time-series data or EKGs?** Read the illustrated [Student Guide](docs/STUDENT_GUIDE.md): EKG basics,
the time-series checklist, how each part of the app works, the AI agent workflow, how real devices differ,
and step-by-step labs (including "how does AFib work?").

## Layout

```
src/signal_gen.py   Gaussian-superposition ECG generator + MIT-BIH (wfdb) loader
src/monitor.py      "Live replay": canvas animation of the analysed window (replay, not real-time processing)
src/dsp.py          Butterworth band-pass, Pan-Tompkins, HRV, QRS/PR/QT estimates, Welch PSD
src/model.py        Random Forest rhythm classifier (5 classes), synthetic training set
src/agent.py        Claude tutor: Pydantic schema + structured output via messages.parse
src/app.py          Streamlit dashboard
src/styles.py       IBM Carbon dark theme (from the /carbon-streamlit skill)
tests/              pytest suite
docs/               Student guide, design spec, figure generator (docs/make_figures.py)
```

Read the code comments: every module explains the maths and the reasoning (Gaussian wave model,
Nyquist, why 0.5-45 Hz, Pan-Tompkins stages, HRV formulas, prompt and structured-output design).

## Configuration

`.env` (see `.env.example`):

- `ANTHROPIC_API_KEY`: required for the tutor
- `ANTHROPIC_MODEL`: optional, defaults to `claude-sonnet-5`

## Tests

```bash
uv run pytest                    # everything (MIT-BIH tests skip themselves if offline)
uv run pytest -m "not network"   # no internet
uv run pytest -m live            # one real Claude API call (needs the key)
```

## Known limitations

- Single lead (Lead II equivalent), 10-30 s windows.
- PR / QRS / QT values are simple heuristics, shown as estimates.
- The classifier is trained on synthetic data by design; see the domain-gap note in the app.
  On the four bundled MIT-BIH records it gets 11 of 12 test slices right. The miss is record 100 at 0 s,
  a normal rhythm with one premature *atrial* beat that the model reads as a PVC: it has no
  atrial-ectopy class. Good discussion material.
