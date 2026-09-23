# CardioCore AI — Implementation Plan

## Context
Build the educational single-screen Streamlit app described in `prd_cardiocore_ai.md`: synthesize or load real EKG time series, filter and detect QRS (Pan-Tompkins), derive HRV/interval features, classify rhythm with a Random Forest, and have a Claude agent return a structured tutoring response (summary, wave breakdown, teaching concept, red flags, quiz). It is a learning/teaching demo, so code must be heavily commented (math, DSP rationale, agent logic per PRD §8).

Current repo state: bare `uv init` skeleton (`main.py` hello-world, empty `README.md`, `pyproject.toml` with `dependencies = []`, `requires-python >=3.12`). Greenfield, so no existing code to reuse.

## Decisions (confirmed with user)
| Topic | Decision |
|---|---|
| Styling | IBM Carbon **dark (g100)** via the `/carbon-streamlit` skill (`styles.py` + `.streamlit/config.toml`), then compacted to fit one screen. Skill templates are light, so override the `THEME` dict: bg `#161616`, panel `#262626`, text `#f4f4f4`/`#c6c6c6`, border `#393939`, primary `#4589ff`; set `base="dark"` in config.toml. |
| Real data (Mode B) | **PhysioNet MIT-BIH via `wfdb`**, not Hugging Face (HF versions are mostly pre-segmented beats, not continuous traces). Fetch a few 30 s slices, cache to `data/` as `.npz` plus `@st.cache_data`. Include the beat annotations so detected peaks can be scored against ground truth. |
| LLM | `claude-sonnet-5`, configurable via `ANTHROPIC_MODEL` in `.env`. The PRD's "Claude 3.5 Sonnet" is outdated. |
| ML training data | Synthetic windows from our own generator, trained on first launch and cached with `joblib`. Show the synthetic-to-real gap on MIT-BIH as a teaching moment. |
| API key | The user supplies it in `.env` (`ANTHROPIC_API_KEY`); never committed. |

## Architecture / files
```
.env.example   .gitignore (ensure .env, data/, models/)   CLAUDE.md (run/test instructions)
pyproject.toml (add deps; pytest pythonpath=["src"])         README.md (quickstart + disclaimer)
.streamlit/config.toml            # Carbon dark theme
src/
  styles.py       # Carbon CSS generator (from skill) + dark THEME + compact/no-scroll overrides
  signal_gen.py   # Gaussian-superposition ECG; AWGN + 0.5 Hz baseline wander; AFib + PVC injection; wfdb loader
  dsp.py          # Butterworth bandpass, Pan-Tompkins, RR/HRV, interval estimates, Welch PSD
  model.py        # feature vector -> RandomForest; synthetic training set; joblib cache
  agent.py        # Pydantic schema + Claude call (messages.parse); graceful offline fallback
  app.py          # Streamlit dashboard (single screen)
tests/            # test_signal_gen.py, test_dsp.py, test_model.py, test_agent.py, test_app.py
```
Delete the leftover root `main.py`. `streamlit run src/app.py` puts `src/` on `sys.path`, so modules use sibling imports.

### signal_gen.py
- Per beat: `a_i * exp(-(t-t_i)^2 / (2σ_i^2))` for P,Q,R,S,T (documented in comments). Beats placed on an RR series: regular with small jitter for NSR; AFib = irregular RR (e.g. lognormal, CV ~0.2–0.3) with the P wave amplitude set to 0; PVC = an early wide, tall beat (large σ_R, inverted T, no P) followed by a compensatory pause.
- Controls: BPM 40–180, SNR in dB (AWGN), baseline wander 0.5 Hz sinusoid, `fs ∈ {250, 500}`, window 10–30 s, seeded RNG for reproducibility. Returns `(t, signal, truth)` where `truth` holds the true beat times and label.
- `load_mitbih(record, start_s, dur_s)`: `wfdb.rdrecord/rdann(pn_dir="mitdb")`, MLII channel, 360 Hz, return signal plus annotated beat samples; cached on disk. Candidate records: 100 (NSR), 119 (PVC/bigeminy), 201 (AFib+PVC), 208. Confirm rhythm content at implementation time by reading the annotations.

### dsp.py
- `bandpass(x, fs, 0.5, 45)`: 4th-order Butterworth as SOS with `sosfiltfilt` (zero phase). Comments cover Nyquist (`fs/2`), why 0.5 Hz (baseline wander) and 45 Hz (EMG/mains), and why zero-phase matters for timing.
- `pan_tompkins(x, fs)`: 5–15 Hz band → derivative → square → moving-window integration (~150 ms) → adaptive signal/noise thresholds with a 200 ms refractory period → refine peak location on the filtered signal. Return R indices plus the intermediate stages so the UI can show them.
- Features: mean HR, mean RR, SDNN, RMSSD, pNN50, RR coefficient of variation, PSD via Welch (LF/HF ratio). Delineation estimates: QRS width (onset/offset by slope threshold), PR and QT via windowed search around each R peak. These are labelled as estimates. `p_wave_present` = P-window energy against a noise floor.
- `score_detection(detected, truth, tol_ms=50)` → sensitivity/PPV, used by tests and by the UI for MIT-BIH.

### model.py
- Classes: NSR, Sinus Bradycardia, Sinus Tachycardia, AFib, PVC (5 classes; the PRD lumps brady/tachy). Features: mean HR, SDNN, RMSSD, pNN50, RR CV, min/max RR ratio, QRS width, fraction of wide-QRS beats, `p_wave_present`, LF/HF.
- Training: ~3–5k synthetic windows with randomized BPM/SNR/parameters, run through the real DSP pipeline so the features match inference. `RandomForestClassifier`, hold-out report, `joblib` dump to `models/`. Retrain if the file is missing.
- `predict(features) -> (label, confidence, per-class probabilities)`. Expose feature importances for the "ECG Physics" tab.

### agent.py
- Pydantic `EKGAnalysisResponse`: `diagnostic_summary`, `wave_breakdown` (P/QRS/T), `teaching_concept`, `clinical_red_flags: list[str]`, `interactive_quiz_question` (`question`, four `options`, `correct_index`, `explanation`).
- Serialize the PRD's payload JSON into the user turn; the system prompt sets the tutor persona, mandates the educational-only disclaimer and no real-patient advice, and states that features are estimates.
- `client.messages.parse(model=…, max_tokens≈4000, output_format=EKGAnalysisResponse, output_config={"effort": "medium"})`. Read `response.parsed_output`. Do not pass `temperature` or `budget_tokens` (rejected on Sonnet 5). Check `stop_reason` for `refusal`/`max_tokens`. Catch the SDK errors most-specific-first (`AuthenticationError`, `RateLimitError`, `APIConnectionError`, `APIStatusError`) and return a friendly message. No key → the panel shows setup instructions and the rest of the app keeps working.
- Cache results in `st.session_state`, keyed by a hash of the feature payload, so a slider tweak doesn't re-bill.
- Provider check done: repo has no non-Anthropic markers.

### app.py (single screen, per PRD §6.2)
- Header row: title + Mode toggle (Synthetic | MIT-BIH). This replaces the PRD's "sidebar toggle" wording, which contradicts its own layout map.
- Left 65%: Plotly chart (raw or filtered overlay, R peaks highlighted, a stage selector for derivative/squared/integrated, zoom/pan, fixed height) → compact controls grid (BPM, SNR, arrhythmia select, fs, window; MIT-BIH record picker in Mode B) → `st.metric` row (HR, PR, QRS, QT, SDNN, RMSSD).
- Right 35%: classification badge with confidence; tabs **Analysis** (Claude output, red flags, interactive quiz via `st.radio` + reveal) and **ECG Physics** (feature matrix table, PSD chart, RF probabilities/importances, detection score in Mode B). A **"Ask Claude tutor"** button triggers the API call, so sliders stay instant and cost is controlled.
- One-line disclaimer footer: educational simulation, not a medical device (PRD §9).
- No-scroll approach: Carbon CSS with tight padding, fixed chart height, and the right panel as a `st.container(key="agent_panel")` given a `vh` height and internal scroll via CSS on `.st-key-agent_panel`. Verified at 1920×1080 and 1366×768.

## Build order (TDD, small commits)
1. Scaffold: `uv add streamlit plotly numpy scipy pandas scikit-learn anthropic pydantic wfdb python-dotenv joblib` and `uv add --dev pytest`. Run `/carbon-streamlit`, apply the dark theme, add `.env.example`, `.gitignore`, `CLAUDE.md`, README, and remove `main.py`. The `datasets` dependency is dropped.
2. `signal_gen` + tests.
3. `dsp` + tests.
4. `model` + tests.
5. `agent` + tests with a mocked client.
6. `app` + `AppTest` smoke test, then visual tuning.
7. Final pass on comments and README.

## Verification
- `uv run pytest`. Targets:
  - clean synthetic 72 BPM → detected HR within ±2 BPM;
  - R-peak sensitivity/PPV ≥ 0.99 at 20 dB SNR, and degrading gracefully at 5 dB;
  - the bandpass removes a 0.3 Hz drift and a 60 Hz tone;
  - AFib RR CV > NSR RR CV;
  - RF hold-out accuracy ≥ ~0.9 on synthetic data;
  - the agent schema round-trips and the offline fallback works with no key.
- MIT-BIH: fetch record 100 and check sensitivity/PPV against the annotations (needs internet once; the test is marked and skippable offline).
- Live agent smoke test (`pytest -m live`) once the user adds the key: one real call must return a valid `EKGAnalysisResponse`.
- `uv run streamlit run src/app.py`: exercise the sliders, arrhythmia injection, mode switch, Ask-Claude button and quiz in the browser. Screenshot at 1920×1080 and 1366×768 to confirm there is no page scroll.

## Notes / risks
- PR/QT/QRS estimation is heuristic. It is shown as "estimate", and the comments explain the limits (single lead, no fiducial-point standard).
- Synthetic-trained RF is expected to underperform on real MIT-BIH data, and the app makes this visible on purpose.
- The first MIT-BIH load needs internet; the cache makes subsequent loads instant.
- After approval, this plan will also be saved as the design spec at `docs/superpowers/specs/2026-09-23-cardiocore-ai-design.md` (brainstorming skill convention) and committed.
