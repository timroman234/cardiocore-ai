"""
app.py -- CardioCore AI: single-screen Streamlit dashboard.

Run:  uv run streamlit run src/app.py

Data flow on every interaction (Streamlit re-runs this script top to bottom):

   controls -> signal_gen (synthetic OR MIT-BIH) -> dsp.analyze (filter, Pan-Tompkins, features)
            -> model.predict (Random Forest)     -> UI
   "Ask Claude tutor" button -> agent.analyze_with_claude (only when clicked, so sliders stay
   instant and API cost stays under the student's control).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import agent  # noqa: E402  (after load_dotenv so the API key is in the environment)
import model  # noqa: E402
from dsp import analyze, score_detection  # noqa: E402
from signal_gen import MITBIH_RECORDS, load_mitbih, synthesize_ecg  # noqa: E402
from styles import CLASS_COLORS, THEME, get_carbon_css  # noqa: E402

st.set_page_config(page_title="CardioCore AI", page_icon="🫀", layout="wide", initial_sidebar_state="collapsed")
st.markdown(get_carbon_css(), unsafe_allow_html=True)

CHART_HEIGHT = 330
RHYTHM_CHOICES = {"Normal sinus": "NSR", "Atrial fibrillation": "AFib", "PVC (premature beats)": "PVC"}


# ----------------------------------------------------------------------------------------------
# Cached heavy lifting
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the rhythm classifier on synthetic ECGs (first launch only, ~1-2 min)...")
def get_model() -> dict:
    return model.load_or_train()


@st.cache_data(show_spinner=False, max_entries=64)
def get_signal(mode, bpm, snr, rhythm, fs, duration, seed, mit_record, mit_start):
    """Generate/load the ECG window and run the DSP pipeline.  Cached: identical settings are free."""
    if mode == "MIT-BIH (real)":
        rec = load_mitbih(mit_record, mit_start, duration)
    else:
        # More noise usually comes with more baseline wander too: tie the two together.
        wander = float(np.interp(snr, [5, 40], [0.35, 0.05]))
        rec = synthesize_ecg(bpm=bpm, duration_s=duration, fs=fs, snr_db=snr, wander_mv=wander,
                             rhythm=rhythm, seed=seed)
    return rec, analyze(rec.signal, rec.fs)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=45, r=10, t=8, b=32),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=THEME["bg_primary"],
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color=THEME["text_secondary"]),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=10)),
        uirevision="keep",   # keep the user's zoom/pan when sliders re-run the script
    )
    fig.update_xaxes(gridcolor=THEME["border"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["border"], zeroline=False)
    return fig


def ecg_figure(rec, a, show_raw: bool, show_truth: bool) -> go.Figure:
    fig = go.Figure()
    if show_raw:
        fig.add_scatter(x=rec.t, y=rec.signal, name="raw", line=dict(color="#6f6f6f", width=1))
    fig.add_scatter(x=rec.t, y=a.filtered, name="filtered 0.5-45 Hz", line=dict(color=THEME["primary"], width=1.6))
    pk = a.peaks
    fig.add_scatter(x=rec.t[pk], y=a.filtered[pk], mode="markers", name="R peaks (Pan-Tompkins)",
                    marker=dict(color=THEME["success"], size=9, symbol="triangle-down"))
    if show_truth and len(rec.beat_times):
        idx = np.clip(np.round(rec.beat_times * rec.fs).astype(int), 0, len(rec.t) - 1)
        fig.add_scatter(x=rec.t[idx], y=a.filtered[idx] * 1.0 + 0.0, mode="markers", name="true beats",
                        marker=dict(color="rgba(0,0,0,0)", size=13, symbol="circle-open",
                                    line=dict(color=THEME["warning"], width=1.5)))
    fig.update_xaxes(title_text="Time (s)", title_standoff=2)
    fig.update_yaxes(title_text="mV", title_standoff=2)
    return _style(fig, CHART_HEIGHT)


def pipeline_figure(rec, a) -> go.Figure:
    """The four Pan-Tompkins stages, stacked, so students can watch the QRS 'stand out'."""
    stages = [("5-15 Hz band", a.pt.band), ("derivative", a.pt.derivative),
              ("squared", a.pt.squared), ("moving-window integral", a.pt.integrated)]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[s for s, _ in stages])
    for i, (name, y) in enumerate(stages, start=1):
        fig.add_scatter(x=rec.t, y=y, name=name, showlegend=False, line=dict(color=THEME["primary"], width=1.2),
                        row=i, col=1)
        fig.add_scatter(x=rec.t[a.peaks], y=y[a.peaks], mode="markers", showlegend=False,
                        marker=dict(color=THEME["success"], size=6, symbol="triangle-down"), row=i, col=1)
    fig.update_annotations(font_size=10, x=0.01, xanchor="left", yshift=-8)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1, title_standoff=2)
    fig.update_layout(margin=dict(l=45, r=10, t=14, b=32))
    return _style(fig, CHART_HEIGHT + 12)


def psd_figure(a) -> go.Figure:
    f, p = a.spectrum_f, a.spectrum_p
    keep = f <= 60
    fig = go.Figure(go.Scatter(x=f[keep], y=10 * np.log10(p[keep] + 1e-12), line=dict(color=THEME["primary"], width=1.4)))
    fig.add_vrect(x0=0.5, x1=45, fillcolor=THEME["primary"], opacity=0.08, line_width=0)
    fig.update_xaxes(title_text="Frequency (Hz)  -  shaded: 0.5-45 Hz pass-band", title_standoff=2)
    fig.update_yaxes(title_text="Power (dB)", title_standoff=2)
    return _style(fig, 180)


def proba_figure(probs: dict[str, float]) -> go.Figure:
    items = sorted(probs.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=[CLASS_COLORS.get(k, THEME["primary"]) for k, _ in items],
                           text=[f"{v:.0%}" for _, v in items], textposition="outside"))
    fig.update_xaxes(range=[0, 1.15], showticklabels=False)
    return _style(fig, 150).update_layout(margin=dict(l=110, r=10, t=4, b=4), showlegend=False)


# ----------------------------------------------------------------------------------------------
# Small HTML helpers
# ----------------------------------------------------------------------------------------------
def card(title: str, body_html: str, kind: str = "") -> None:
    st.markdown(f'<div class="card {kind}"><div class="c-title">{title}</div><div class="c-body">{body_html}</div></div>',
                unsafe_allow_html=True)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v, unit="", digits=0) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}{unit}"


# ----------------------------------------------------------------------------------------------
# Header: title, chart view, data mode
# ----------------------------------------------------------------------------------------------
st.session_state.setdefault("seed", 42)
st.session_state.setdefault("lessons", {})

h1, h2, h3 = st.columns([0.42, 0.30, 0.28], vertical_alignment="center")
with h1:
    st.markdown("# CardioCore AI")
    st.markdown('<div class="subtitle">Time-series cardiac analyzer &amp; agentic diagnostic tutor</div>', unsafe_allow_html=True)
with h2:
    view = st.segmented_control("View", ["ECG", "Pan-Tompkins stages"], default="ECG",
                                label_visibility="collapsed", key="view") or "ECG"
with h3:
    mode = st.segmented_control("Data source", ["Synthetic", "MIT-BIH (real)"], default="Synthetic",
                                label_visibility="collapsed", key="mode") or "Synthetic"

bundle = get_model()

left, right = st.columns([0.65, 0.35])

# ----------------------------------------------------------------------------------------------
# LEFT: controls (rendered first so their values exist), chart, metrics
# ----------------------------------------------------------------------------------------------
with left:
    chart_slot = st.container()

    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 0.7], vertical_alignment="bottom")
    if mode == "Synthetic":
        bpm = c1.slider("Heart rate (bpm)", 40, 180, 72, key="bpm")
        snr = c2.slider("Signal-to-noise (dB)", 5, 40, 30, key="snr",
                        help="Lower = noisier. Try 5-10 dB to break the peak detector.")
        rhythm_name = c3.selectbox("Inject arrhythmia", list(RHYTHM_CHOICES), key="rhythm")
        rhythm = RHYTHM_CHOICES[rhythm_name]
        with c4.popover("More"):
            fs = st.select_slider("Sampling rate (Hz)", [250, 500], 250, key="fs")
            duration = st.select_slider("Window (s)", [10, 20, 30], 10, key="dur")
            if st.button("New random noise"):
                st.session_state["seed"] += 1
        mit_record, mit_start = "100", 0
    else:
        bpm, snr, rhythm, fs = 72, 30, "NSR", 360
        mit_record = c1.selectbox("Record", list(MITBIH_RECORDS), key="rec",
                                  format_func=lambda r: f"{r} - {MITBIH_RECORDS[r]['desc']}")
        mit_start = c2.slider("Start (s)", 0, 900, 0, step=30, key="start")
        duration = c3.select_slider("Window (s)", [10, 20, 30], 10, key="dur_mit")
        c4.caption("360 Hz, MLII")

    try:
        rec, a = get_signal(mode, bpm, snr, rhythm, fs, duration, st.session_state["seed"], mit_record, mit_start)
    except Exception as exc:  # e.g. PhysioNet unreachable on first download
        st.error(f"Could not load MIT-BIH data ({exc}). The first download needs internet access; "
                 "switch to Synthetic to keep going.")
        st.stop()

    with chart_slot:
        if view == "ECG":
            show_raw = st.session_state.get("show_raw", True)
            show_truth = st.session_state.get("show_truth", False)
            st.plotly_chart(ecg_figure(rec, a, show_raw, show_truth), width="stretch",
                            config=dict(displaylogo=False, modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"]))
        else:
            st.plotly_chart(pipeline_figure(rec, a), width="stretch", config=dict(displaylogo=False))
        if view == "ECG":
            t1, t2, t3 = st.columns([0.2, 0.2, 0.6], vertical_alignment="center")
            t1.checkbox("Raw", value=True, key="show_raw")
            t2.checkbox("True beats", value=False, key="show_truth")
            t3.caption(rec.description)

    f = a.features
    m = st.columns(6)
    m[0].metric("Heart rate", fmt(f["heart_rate_bpm"], " bpm") if f["valid"] else "n/a")
    m[1].metric("PR (est.)", fmt(f["pr_interval_ms"], " ms"))
    m[2].metric("QRS (est.)", fmt(f["qrs_duration_ms"], " ms"))
    m[3].metric("QT (est.)", fmt(f["qt_interval_ms"], " ms"))
    m[4].metric("SDNN", fmt(f["sdnn_ms"], " ms", 1))
    m[5].metric("RMSSD", fmt(f["rmssd_ms"], " ms", 1))

# ----------------------------------------------------------------------------------------------
# RIGHT: classification badge + tabs (Analysis | ECG Physics)
# ----------------------------------------------------------------------------------------------
with right:
    if not f["valid"]:
        st.warning("Fewer than 3 beats detected in this window: not enough to analyse.")
        st.stop()

    pred = model.predict(f, bundle)
    color = CLASS_COLORS.get(pred.label, THEME["primary"])
    truth_line = f"annotated / true label: {rec.label}" if rec.label else ""
    st.markdown(f'<div class="badge" style="--c:{color}"><div class="b-title">{pred.label.upper()} '
                f'({pred.confidence:.0%})</div><div class="b-sub">Random Forest verdict &middot; {truth_line}</div></div>',
                unsafe_allow_html=True)

    panel = st.container(key="agent_panel")
    with panel:
        tab_analysis, tab_physics = st.tabs(["Analysis", "ECG Physics"])

        # ---------------- Tab 1: AI tutor ----------------
        with tab_analysis:
            payload = agent.build_payload(f, pred, source=rec.source)
            key = agent.payload_key(payload)
            lessons: dict = st.session_state["lessons"]

            if st.button("Ask Claude tutor", type="primary", key="ask"):
                with st.spinner("Claude is preparing your lesson..."):
                    lessons[key] = agent.analyze_with_claude(payload)
                st.session_state["last_key"] = key

            shown_key = key if key in lessons else st.session_state.get("last_key")
            result = lessons.get(shown_key)
            if result is None:
                st.markdown('<div class="card"><div class="c-title">AI tutor</div><div class="c-body">'
                            "Press <b>Ask Claude tutor</b> to get an explanation, red flags and a quiz "
                            "for the signal on screen. The tutor only sees the numbers measured by the "
                            "DSP engine and the classifier, never the raw waveform.</div></div>",
                            unsafe_allow_html=True)
                if not agent.has_api_key():
                    st.caption("No ANTHROPIC_API_KEY set: copy .env.example to .env and add your key.")
            elif result.error:
                st.error(result.error)
            else:
                if shown_key != key:
                    st.info("This lesson is for an earlier signal. Press the button to refresh it.")
                r = result.analysis
                card("Diagnostic summary", esc(r.diagnostic_summary))
                wb = r.wave_breakdown
                card("Wave breakdown",
                     f'<div class="wave"><b>P</b> {esc(wb.p_wave)}</div>'
                     f'<div class="wave"><b>QRS</b> {esc(wb.qrs_complex)}</div>'
                     f'<div class="wave"><b>T</b> {esc(wb.t_wave)}</div>')
                card("Physiologic mechanism", esc(r.teaching_concept))
                card("Clinical red flags", "<ul style='margin:0;padding-left:1.1rem'>"
                     + "".join(f"<li>{esc(x)}</li>" for x in r.clinical_red_flags) + "</ul>", "flag")

                q = r.interactive_quiz_question
                card("Self-test", esc(q.question), "quiz")
                choice = st.radio("Your answer", q.options, index=None, key=f"quiz_{shown_key}",
                                  label_visibility="collapsed")
                if choice is not None:
                    if q.options.index(choice) == q.correct_index:
                        st.success("Correct! " + q.explanation)
                    else:
                        st.error("Not quite. " + q.explanation)
                st.caption(f"Model: {result.model}")

        # ---------------- Tab 2: physics / features ----------------
        with tab_physics:
            rows = [
                ("Heart rate", fmt(f["heart_rate_bpm"], " bpm"), "60 / mean R-R interval"),
                ("Mean R-R", fmt(f["mean_rr_ms"], " ms"), "time between beats"),
                ("SDNN", fmt(f["sdnn_ms"], " ms", 1), "overall R-R variability"),
                ("RMSSD", fmt(f["rmssd_ms"], " ms", 1), "beat-to-beat variability"),
                ("pNN50", fmt(f["pnn50"], " %", 0), "successive R-R diffs > 50 ms"),
                ("R-R CV", fmt(f["rr_cv"], "", 3), "SDNN / mean R-R (AFib is high)"),
                ("QRS width", fmt(f["qrs_duration_ms"], " ms"), "ventricular depolarisation (est.)"),
                ("Wide-QRS beats", fmt(100 * f["wide_qrs_fraction"], " %"), "share >= 120 ms (PVC sign)"),
                ("P wave", "present" if f["p_wave_present"] else "absent", "atrial activity before QRS"),
                ("PR interval", fmt(f["pr_interval_ms"], " ms"), "atria -> ventricles (est.)"),
                ("QT interval", fmt(f["qt_interval_ms"], " ms"), "depolarisation + repolarisation (est.)"),
                ("LF/HF", fmt(f["lf_hf"], "", 2), "needs a window >= 20 s"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Feature", "Value", "Meaning"]), hide_index=True,
                         width="stretch", height=250)

            s = score_detection(a.peaks / rec.fs, rec.beat_times, duration_s=rec.t[-1] + 1 / rec.fs)
            st.caption(f"Pan-Tompkins vs {'annotated' if rec.source == 'mitbih' else 'true'} beats: "
                       f"sensitivity {s['sensitivity']:.1%} - PPV {s['ppv']:.1%} "
                       f"({s['tp']} hit, {s['fn']} missed, {s['fp']} false)")

            st.markdown("**Power spectrum (FFT / Welch)**")
            st.plotly_chart(psd_figure(a), width="stretch", config=dict(displayModeBar=False))
            st.markdown("**Classifier vote share**")
            st.plotly_chart(proba_figure(pred.probabilities), width="stretch", config=dict(displayModeBar=False))
            st.caption(f"Random Forest trained on {bundle['n_train']} synthetic windows, hold-out accuracy "
                       f"{bundle['accuracy']:.0%} on synthetic data.")
            if rec.source == "mitbih":
                agree = "agrees with" if pred.label == rec.label else "DISAGREES with"
                st.info(f"Domain gap: on this real recording the synthetic-trained model {agree} the "
                        f"annotation-derived label ({rec.label}). Real ECGs are messier than our model of them.")
            top = sorted(bundle["importances"].items(), key=lambda kv: -kv[1])[:4]
            st.caption("Most informative features: " + ", ".join(f"{k} ({v:.0%})" for k, v in top))

st.markdown('<div class="disclaimer">Educational simulation only. Not a medical device; never use for real '
            'diagnosis or triage.</div>', unsafe_allow_html=True)
