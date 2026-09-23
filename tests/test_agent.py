"""Tests for the Claude tutor (src/agent.py). Network-free except the `live` test."""
import os
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest
from dotenv import load_dotenv

import agent
from agent import EKGAnalysisResponse, analyze_with_claude, build_payload, payload_key
from model import Prediction

load_dotenv()  # so the opt-in `live` test can see ANTHROPIC_API_KEY from .env

FEATURES = dict(heart_rate_bpm=112.3, mean_rr_ms=535.0, sdnn_ms=14.24, rmssd_ms=12.0, pnn50=3.0,
                qrs_duration_ms=110.0, wide_qrs_fraction=0.0, pr_interval_ms=float("nan"),
                qt_interval_ms=310.0, p_wave_present=True)
PRED = Prediction("Sinus Tachycardia", 0.94, {"Sinus Tachycardia": 0.94})

VALID = EKGAnalysisResponse.model_validate(dict(
    diagnostic_summary="Fast regular rhythm.",
    wave_breakdown=dict(p_wave="Present.", qrs_complex="Narrow.", t_wave="Upright."),
    teaching_concept="Sympathetic drive shortens diastole.",
    clinical_red_flags=["Persistent rate > 150"],
    interactive_quiz_question=dict(question="Q?", options=["a", "b", "c", "d"], correct_index=2, explanation="Because."),
))


class FakeClient:
    """Stands in for anthropic.Anthropic: records the call and returns/raises what we script."""
    def __init__(self, result=None, exc=None):
        self.calls, self._result, self._exc = [], result, exc
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return self._result


def _resp(stop="end_turn", parsed=VALID):
    return SimpleNamespace(stop_reason=stop, parsed_output=parsed, model="claude-sonnet-5")


def test_payload_matches_prd_shape_and_nan_becomes_null():
    p = build_payload(FEATURES, PRED)
    assert p["heart_rate_bpm"] == 112 and p["mean_rr_ms"] == 535 and p["sdnn_ms"] == 14.2
    assert p["ml_prediction"] == "Sinus Tachycardia" and p["ml_confidence"] == 0.94
    assert p["p_wave_present"] is True and p["pr_interval_ms"] is None


def test_payload_key_is_stable_and_sensitive():
    p = build_payload(FEATURES, PRED)
    assert payload_key(p) == payload_key(dict(reversed(list(p.items()))))
    assert payload_key(p) != payload_key({**p, "heart_rate_bpm": 113})


def test_schema_round_trips():
    assert EKGAnalysisResponse.model_validate_json(VALID.model_dump_json()) == VALID


def test_success_path_sends_schema_and_payload():
    fake = FakeClient(_resp())
    r = analyze_with_claude(build_payload(FEATURES, PRED), client=fake)
    assert r.error is None and r.analysis == VALID
    call = fake.calls[0]
    assert call["output_format"] is EKGAnalysisResponse
    assert "temperature" not in call and "thinking" not in call     # rejected/unneeded on Sonnet 5
    assert '"ml_prediction": "Sinus Tachycardia"' in call["messages"][0]["content"]
    assert call["model"] == "claude-sonnet-5"


def test_model_is_configurable_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    fake = FakeClient(_resp())
    analyze_with_claude({}, client=fake)
    assert fake.calls[0]["model"] == "claude-haiku-4-5"


def test_no_api_key_gives_setup_message_without_calling_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = analyze_with_claude({})
    assert r.analysis is None and ".env" in r.error


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_unusable_stop_reasons_become_errors(stop):
    r = analyze_with_claude({}, client=FakeClient(_resp(stop=stop)))
    assert r.analysis is None and r.error


def test_missing_parsed_output_is_an_error():
    assert analyze_with_claude({}, client=FakeClient(_resp(parsed=None))).error


def test_api_errors_are_translated_not_raised():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    def err(cls, status):
        return cls("boom", response=httpx.Response(status, request=req), body=None)
    cases = [
        (err(anthropic.AuthenticationError, 401), "rejected the API key"),
        (err(anthropic.RateLimitError, 429), "Rate limited"),
        (err(anthropic.NotFoundError, 404), "not found"),
        (anthropic.APIConnectionError(request=req), "internet"),
    ]
    for exc, text in cases:
        r = analyze_with_claude({}, client=FakeClient(exc=exc))
        assert r.analysis is None and text in r.error


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_call_returns_valid_structured_lesson():
    r = analyze_with_claude(build_payload(FEATURES, PRED))
    assert r.error is None, r.error
    assert len(r.analysis.interactive_quiz_question.options) == 4
    assert 0 <= r.analysis.interactive_quiz_question.correct_index <= 3
