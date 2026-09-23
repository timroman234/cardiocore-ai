# Product Requirement Document (PRD)

## CardioCore AI: Interactive Time-Series EKG Signal Processing & Agentic Diagnostic Workspace

---

## 1. Project Overview & Objectives

### 1.1 Purpose
**CardioCore AI** is a single-screen, highly interactive educational web application designed to demonstrate time-series signal processing, digital signal filtering (DSP), machine learning signal classification, and LLM-driven medical triage. 

The application synthesizes multi-beat EKG time-series waveforms mathematically or loads benchmark clinical time-series data from PhysioNet (MIT-BIH Arrhythmia Dataset). It runs real-time peak detection, calculates Heart Rate Variability (HRV) and R-R intervals, applies 1D feature extraction, and sends extracted diagnostic metrics to an Anthropic Claude agent for structured clinical summaries and interactive tutoring.

### 1.2 Target Audience
* **Primary:** Medical students, biomedical engineering students, and software developers learning biomedical time-series processing, DSP algorithms, and AI agent orchestration.
* **Core Goal:** Provide a transparent, "under-the-hood" learning workspace where users can manipulate continuous time-series signal physics, observe automated DSP peak detection, evaluate classification metrics, and receive structured AI tutoring.

### 1.3 Key Technical Directives
* **Development Environment:** VS Code with **Claude Code CLI**.
* **Language & Package Manager:** Python 3.11+ managed via **`uv`**.
* **UI Framework:** **Streamlit** styled with custom dark CSS to fit entirely on a single non-scrolling screen (dashboard layout).
* **AI & Agentic Framework:** **Anthropic Claude API** (`anthropic` Python SDK) using structured output and Pydantic schema validation.
* **Code Standard:** Heavily annotated code blocks explaining continuous time-series concepts (e.g., sampling rate $F_s$, Nyquist frequency, Butterworth filtering, Fourier transforms, Pan-Tompkins QRS detection, and Gaussian wave superposition).

---

## 2. System Architecture & Component Design

```
                     +---------------------------------------+
                     |          STREAMLIT DASHBOARD          |
                     |  (Single-Screen Modern Dark Layout)   |
                     +-------------------+-------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      [1. Data Ingestion]                                [2. DSP Engine]
   +-----------------------+                         +-----------------------+
   | Synthetic Wave Engine |                         | - Bandpass Filter     |
   | (Gaussian Superpos.)  |                         | - Pan-Tompkins QRS    |
   |         OR            |                         | - RR Interval / BPM   |
   | HuggingFace MIT-BIH   |                         | - Feature Extraction  |
   +-----------+-----------+                         +-----------+-----------+
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                              [3. Dual AI Pipeline]
                                         |
             +---------------------------+---------------------------+
             |                                                       |
   [3a. ML Rhythm Classifier]                             [3b. Agentic Medical Tutor]
   - Scikit-Learn (Random Forest)                         - Anthropic API (Claude 3.5 Sonnet)
   - Classifies: Normal, AFib,                            - Inputs: Extracted Features + Signal Stats
     PVC, Bradycardia, Tachycardia                        - Output: Pydantic Schema (Explanation,
                                                            Clinical Red Flags, Teaching Quiz)
```

---

## 3. Time-Series Data Strategy & Ingestion

The application supports two distinct operational modes via a top sidebar toggle:

### Mode A: Programmatic Synthetic Generator (Default)
* **Mechanism:** Uses a mathematical Gaussian superposition model to synthesize continuous single-lead EKG time-series arrays.
* **Math Formulation:**
  Each beat is synthesized as the sum of five distinct deflection waves ($P, Q, R, S, T$):
  $$ECG(t) = \sum_{i \in \{P,Q,R,S,T\}} a_i \cdot \exp\left(-\frac{(t - t_i)^2}{2\sigma_i^2}\right)$$
* **Controllable Physics Parameters:**
  * **Heart Rate (BPM):** Range $40 - 180 \text{ BPM}$.
  * **Signal-to-Noise Ratio (SNR):** Additive White Gaussian Noise (AWGN) and low-frequency baseline wander ($0.5 \text{ Hz}$ breathing artifact).
  * **Arrhythmia Injection:** Toggle synthetic Atrial Fibrillation (irregular R-R intervals + missing P-waves) or Premature Ventricular Contractions (broad QRS spikes followed by compensatory pauses).
* **Sampling Rate:** $250 \text{ Hz}$ or $500 \text{ Hz}$ (standard digital clinical resolution).

### Mode B: Hugging Face Benchmark Dataset
* **Source:** `datasets` library fetching `mitbih` (MIT-BIH Arrhythmia Database).
* **Implementation:** Cache local slices using `@st.cache_data` to ensure zero runtime latency, allowing direct comparison against real patient continuous traces.

---

## 4. Signal Processing & Feature Extraction Engine (DSP)

The backend DSP module (`src/dsp.py`) implements and documents the following time-series algorithms:

1. **Filtering & Preprocessing:**
   * **Bandpass Butterworth Filter ($0.5 \text{ Hz} - 45 \text{ Hz}$):** Removes high-frequency muscle tremor noise ($> 45 \text{ Hz}$) and low-frequency baseline wander ($< 0.5 \text{ Hz}$).
2. **Pan-Tompkins QRS Detection Algorithm:**
   * Step 1: Derivative filter to highlight steep QRS slopes.
   * Step 2: Squaring function to amplify high-frequency cardiac spikes.
   * Step 3: Moving-window integration to extract peak boundaries.
   * Step 4: Adaptive thresholding to detect R-peaks.
3. **Derived Diagnostic Features:**
   * **Mean Heart Rate (BPM)** and **Heart Rate Variability ($\text{SDNN} / \text{RMSSD}$)**.
   * **PR Interval**, **QRS Duration**, and **QT Interval** estimates.
   * **Power Spectral Density (PSD)** via Fast Fourier Transform (FFT).

---

## 5. Machine Learning & Agentic Layer

### 5.1 ML Classifier Module (`src/model.py`)
* **Model:** Lightweight `RandomForestClassifier` trained on derived time-domain and frequency-domain feature arrays.
* **Output Classes:**
  1. *Normal Sinus Rhythm (NSR)*
  2. *Atrial Fibrillation (AFib)*
  3. *Premature Ventricular Contraction (PVC)*
  4. *Sinus Bradycardia / Tachycardia*

### 5.2 Agentic Teaching Assistant (`src/agent.py`)
* **LLM Engine:** Anthropic Claude 3.5 Sonnet (`anthropic` SDK).
* **Agent Architecture:** Structured System Prompting with **Pydantic Validation**.
* **Input Payload to LLM:**
  ```json
  {
    "heart_rate_bpm": 112,
    "mean_rr_ms": 535,
    "sdnn_ms": 14.2,
    "qrs_duration_ms": 110,
    "ml_prediction": "Sinus Tachycardia",
    "ml_confidence": 0.94,
    "p_wave_present": true
  }
  ```
* **Structured Output Schema (`EKGAnalysisResponse`):**
  * `diagnostic_summary`: Clear, concise interpretation of the rhythm.
  * `wave_breakdown`: Key observations on P wave, QRS complex, and T wave.
  * `teaching_concept`: Short educational lesson explaining the underlying electrophysiology.
  * `clinical_red_flags`: Warning signs a clinician would evaluate.
  * `interactive_quiz_question`: Multiple-choice question testing the user on this waveform.

---

## 6. UI/UX Specifications (Single-Screen Dashboard Layout)

### 6.1 Design Goal: "Single-Screen Compact Viewport"
* Inject custom CSS to minimize default Streamlit padding (`top-padding: 1rem`, compact margins).
* Prevent page scrolling by organizing components into responsive grids and side-by-side columns.

### 6.2 Screen Layout Map

```
+---------------------------------------------------------------------------------------------------+
| HEADER: CardioCore AI - Time-Series Cardiac Analyzer & Diagnostic Engine    [Mode: Synthetic | HF ] |
+-------------------------------------------------------------+-------------------------------------+
| LEFT COLUMN (Width: 65%) - Time-Series Visualizer           | RIGHT COLUMN (Width: 35%)           |
|                                                             | AI Agent Diagnostic Panel           |
| +---------------------------------------------------------+ |                                     |
| | Interactive Plotly Chart (Dynamic EKG Plot)             | | [ Primary Classification Badge ]   |
| | - Synchronized zoom/pan                                 | | e.g. "ATRIAL FIBRILLATION (92%)"  |
| | - Color-highlighted R-Peaks (Pan-Tompkins output)       | |                                     |
| +---------------------------------------------------------+ | [ Tabs: Analysis | ECG Physics ]  |
|                                                             |                                     |
| +---------------------------------------------------------+ | Tab 1: AI Tutor Explanation         |
| | Interactive Signal Controls (Expandable/Compact Grid)   | | - Physiologic Mechanism           |
| | [ BPM Slider ] [ Noise Level ] [ Inject Arrhythmia ]    | | - Clinical Red Flags              |
| +---------------------------------------------------------+ | - Student Self-Test Quiz            |
|                                                             |                                     |
| +---------------------------------------------------------+ | Tab 2: Feature Matrix Table         |
| | DSP Metrics Row (Compact St.Metrics)                    | | - HR, QRS Width, RR Variability   |
| | [ Heart Rate: 72 bpm ] [ PR: 160ms ] [ QRS: 90ms ]      | |                                     |
+-------------------------------------------------------------+-------------------------------------+
```

---

## 7. Project Structure & Setup Instructions

### 7.1 Directory Layout
```text
cardiocore-ai/
├── .env.example               # Template for ANTHROPIC_API_KEY
├── pyproject.toml             # Managed by uv
├── README.md                  # Quickstart guide
├── CLAUDE.md                  # Execution instructions for Claude Code
└── src/
    ├── __init__.py
    ├── app.py                 # Main Streamlit UI entrypoint
    ├── signal_gen.py          # Math signal generator & HF dataset loader
    ├── dsp.py                 # Time-series filtering & Pan-Tompkins QRS
    ├── model.py               # ML rhythm classifier
    └── agent.py               # Anthropic Claude Agent with structured output
```

### 7.2 Dependencies (`pyproject.toml`)
Managed via `uv`:
* `streamlit` (UI)
* `plotly` (Interactive time-series charting)
* `numpy` & `scipy` (Signal processing & Gaussian math)
* `pandas` (Data manipulation)
* `scikit-learn` (Machine learning classification)
* `anthropic` (Claude API client)
* `pydantic` (Data validation)
* `datasets` (Hugging Face datasets for MIT-BIH)
* `python-dotenv` (Environment management)

---

## 8. Educational Code Commenting Standards

Every file created MUST adhere to these documentation rules:
1. **Math Formula Comments:** Write mathematical equations in clear comment blocks prior to execution (e.g., explaining Gaussian PDF formula used for wave synthesis $f(t) = A \cdot \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)$).
2. **DSP Rationale:** Explain *why* specific cutoff frequencies ($0.5 \text{ Hz} - 45 \text{ Hz}$) are used for medical signal filtering and how sampling rate $F_s$ affects Nyquist boundaries.
3. **Agent Logic Comments:** Annotate prompt structure, context injection, metric serialization, and structured output extraction.

---

## 9. System Limitations & Non-Production Disclaimers

1. **Educational Simulation Only (NOT a Medical Device):** Designed strictly for academic demonstration and software teaching. It is NOT FDA-cleared and must never be used for real-life patient triage or clinical diagnosis.
2. **Single-Lead Simplification:** Demonstrates single-lead vector calculations (Lead II equivalent) rather than full 12-lead vector reconstruction.
3. **Synthetic Artifacts:** Simulated waveforms approximate cardiac vectors using mathematical models and do not capture subtle cardiac pathologies such as electrolyte imbalances or localized ischemia.
4. **In-Memory Windowing:** Processed time-series frames are capped at 10–30 second windows to maintain real-time Streamlit UI responsiveness.

---

## 10. Execution Blueprint for Claude Code

When starting this project in VS Code with Claude Code, run the following sequence:

1. **Initialize Project with UV:**
   ```bash
   uv init cardiocore-ai
   cd cardiocore-ai
   uv add streamlit plotly numpy scipy pandas scikit-learn anthropic pydantic datasets python-dotenv
   ```

2. **Set Environment:**
   Create `.env` containing `ANTHROPIC_API_KEY=your_key_here`.

3. **Run Application:**
   ```bash
   uv run streamlit run src/app.py
   ```