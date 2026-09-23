"""
agent.py -- The agentic medical tutor (PRD section 5.2), powered by the Anthropic Claude API.

What "agentic" means here: the app does not just show numbers.  It hands the *measured* signal
features plus the ML model's opinion to an LLM that plays a cardiology teaching assistant, and gets
back a structured lesson -- summary, wave-by-wave breakdown, the physiology behind it, red flags a
clinician would look for, and a quiz question for the student.

Three design ideas worth studying in this file
----------------------------------------------
1. CONTEXT INJECTION.  The LLM never sees the waveform.  It sees a compact JSON "fact sheet" made by
   our DSP + ML code (`build_payload`).  We do the numeric work in code (cheap, exact,
   reproducible) and use the LLM for what it is good at: explaining and teaching.
2. STRUCTURED OUTPUT.  We describe the answer as a Pydantic class (`EKGAnalysisResponse`).  The SDK
   turns it into a JSON Schema, the API constrains the model to produce JSON matching that
   schema, and `messages.parse()` validates the reply back into a Python object.  No fragile
   string parsing.
3. GUARDRAILS.  The system prompt says "educational simulation, not medical advice", and the code
   handles missing keys, rate limits, refusals and truncation so the UI degrades gracefully.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-sonnet-5"


# ----------------------------------------------------------------------------------------------
# The structured answer we want back
# ----------------------------------------------------------------------------------------------
class WaveBreakdown(BaseModel):
    p_wave: str = Field(description="Observations about the P wave (atrial depolarisation).")
    qrs_complex: str = Field(description="Observations about the QRS complex (ventricular depolarisation).")
    t_wave: str = Field(description="Observations about the T wave (ventricular repolarisation).")


class QuizQuestion(BaseModel):
    question: str = Field(description="A multiple-choice question testing understanding of THIS waveform.")
    options: list[str] = Field(description="Exactly four answer options, without letter prefixes.")
    correct_index: int = Field(description="Zero-based index (0-3) of the correct option.")
    explanation: str = Field(description="Why the correct answer is right and the others are not.")


class EKGAnalysisResponse(BaseModel):
    """PRD schema: the tutor's structured lesson."""

    diagnostic_summary: str = Field(description="Clear, concise interpretation of the rhythm (2-4 sentences).")
    wave_breakdown: WaveBreakdown
    teaching_concept: str = Field(description="Short lesson on the underlying cardiac electrophysiology.")
    clinical_red_flags: list[str] = Field(description="Warning signs a clinician would evaluate for this pattern.")
    interactive_quiz_question: QuizQuestion


# ----------------------------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are CardioCore Tutor, a friendly cardiology and biomedical-signal-processing teaching assistant \
inside an educational demo app. Your students are medical students, biomedical-engineering students \
and software developers.

You receive a JSON fact sheet computed from a single-lead ECG window (Lead II equivalent). The values \
come from automatic signal processing (Butterworth filtering, Pan-Tompkins QRS detection) and a \
Random Forest rhythm classifier trained on synthetic data. Interval values (PR, QRS, QT) are rough \
estimates, and the classifier can be wrong, especially on real recordings. Say so when the numbers \
look inconsistent with the classifier's label, and use that as a teaching moment rather than \
hiding it.

Rules:
- Base every statement on the fact sheet. Do not invent measurements. If a value is null it was \
not measurable; say that.
- This is an educational simulation, NOT a medical device. Never give personal medical advice or \
tell anyone how to treat a real patient.
- Explain the mechanism (why the waveform looks this way), and connect it to the signal-processing \
numbers where useful (e.g. what a high RMSSD means).
- The quiz must be answerable from your own teaching content, have four plausible options, and \
exactly one correct answer.
- Keep each field concise; the results are shown in a small dashboard panel.
"""


def build_payload(features: dict, prediction, source: str = "synthetic") -> dict:
    """
    Serialise the metrics into the JSON fact sheet described in PRD 5.2.

    NaN/None become null (JSON has no NaN) and numbers are rounded, which also makes the payload
    stable enough to hash for caching.
    """
    def num(key, digits=1):
        v = features.get(key)
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits)

    return {
        "heart_rate_bpm": num("heart_rate_bpm", 0),
        "mean_rr_ms": num("mean_rr_ms", 0),
        "sdnn_ms": num("sdnn_ms"),
        "rmssd_ms": num("rmssd_ms"),
        "pnn50_percent": num("pnn50"),
        "qrs_duration_ms": num("qrs_duration_ms", 0),
        "wide_qrs_beat_fraction": num("wide_qrs_fraction", 2),
        "pr_interval_ms": num("pr_interval_ms", 0),
        "qt_interval_ms": num("qt_interval_ms", 0),
        "p_wave_present": bool(features.get("p_wave_present", False)),
        "ml_prediction": prediction.label,
        "ml_confidence": round(prediction.confidence, 2),
        "signal_source": source,
    }


def payload_key(payload: dict) -> str:
    """Stable hash of the payload: same measurements -> same key -> reuse the cached answer."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Calling Claude
# ----------------------------------------------------------------------------------------------
@dataclass
class AgentResult:
    analysis: EKGAnalysisResponse | None = None
    error: str | None = None        # human-readable, safe to show in the UI
    model: str | None = None


def has_api_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


def analyze_with_claude(payload: dict, client: anthropic.Anthropic | None = None,
                        model: str | None = None) -> AgentResult:
    """
    Ask Claude to teach from `payload`.  Never raises: problems come back in `AgentResult.error`.

    Request anatomy:
      system         the tutor persona + rules (stable text)
      messages       one user turn containing the JSON fact sheet
      output_format  our Pydantic class -> JSON Schema -> constrained decoding
      effort=medium  reasoning depth vs latency/cost.  A live classroom demo wants a few seconds,
                     not a deep think.  (Sonnet 5 always uses adaptive thinking; sampling
                     parameters like `temperature` are not accepted on this model family.)
    """
    model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL
    if client is None:
        if not has_api_key():
            return AgentResult(error="No ANTHROPIC_API_KEY found. Copy .env.example to .env, add your "
                                     "key, and restart the app.")
        client = anthropic.Anthropic(timeout=90.0)  # SDK retries 429/5xx twice by default

    user_turn = ("Here is the fact sheet for the ECG window currently on screen:\n\n"
                 + json.dumps(payload, indent=2)
                 + "\n\nWrite the lesson for the student.")
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_turn}],
            output_format=EKGAnalysisResponse,
            output_config={"effort": "medium"},
        )
    # Most specific first: retryable/transient errors are reported differently from setup errors.
    except anthropic.AuthenticationError:
        return AgentResult(error="Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.PermissionDeniedError:
        return AgentResult(error="This API key is not allowed to use that model.")
    except anthropic.NotFoundError:
        return AgentResult(error=f"Model '{model}' was not found. Set ANTHROPIC_MODEL in .env.")
    except anthropic.RateLimitError:
        return AgentResult(error="Rate limited by the Anthropic API. Wait a moment and try again.")
    except anthropic.APIConnectionError:
        return AgentResult(error="Could not reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as exc:
        return AgentResult(error=f"Anthropic API error ({exc.status_code}): {exc.message}")
    except Exception as exc:  # e.g. the reply failed Pydantic validation
        return AgentResult(error=f"Could not read the tutor's reply: {exc}")

    # A 200 response can still be unusable: check why the model stopped before trusting content.
    if response.stop_reason == "refusal":
        return AgentResult(error="The model declined to answer this request.")
    if response.stop_reason == "max_tokens":
        return AgentResult(error="The tutor's answer was cut off. Please try again.")
    if response.parsed_output is None:
        return AgentResult(error="The tutor returned no structured answer. Please try again.")
    return AgentResult(analysis=response.parsed_output, model=response.model)
