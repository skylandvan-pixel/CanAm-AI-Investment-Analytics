"""Unit tests for core.committee_context.build_jev_context -- pure shaping,
no I/O, no committee/AI imports. See tests/test_app_jev_integration.py for the
end-to-end wiring tests (Jev never changes the committee's decision)."""
import pytest

from core.committee_context import build_jev_context
from core.jev_models import JevResponse, JevResult

VALID_RESPONSE = {
    "model": "jev-test-version",
    "answers": {
        "market_bullish": {"type": "noul", "noul": .9},
        "market_regime": {
            "type": "choice", "choice": "BULL", "confidence": .85,
            "probabilities": {"BULL": .85, "NEUTRAL": .1, "BEAR": .05},
        },
    },
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def test_ok_status_produces_available_context_with_summary():
    result = JevResult(status="ok", response=JevResponse.model_validate(VALID_RESPONSE), attempts=1, latency_ms=10.0)
    ctx = build_jev_context(result)
    assert ctx["available"] is True
    assert ctx["status"] == "ok"
    assert ctx["summary"] == "偏乐观｜Confidence 85%"
    assert ctx["confidence"] == pytest.approx(.85)
    assert ctx["signals"]["market_regime_choice"] == "BULL"
    assert ctx["signals"]["market_bullish_probability"] == pytest.approx(.9)
    assert ctx["reason"] is None


@pytest.mark.parametrize("choice,label", [("BULL", "偏乐观"), ("NEUTRAL", "中性"), ("BEAR", "偏谨慎")])
def test_regime_label_mapping(choice, label):
    payload = {**VALID_RESPONSE, "answers": {**VALID_RESPONSE["answers"], "market_regime": {
        "type": "choice", "choice": choice, "confidence": .6,
        "probabilities": {"BULL": .6 if choice == "BULL" else .2, "NEUTRAL": .6 if choice == "NEUTRAL" else .2,
                          "BEAR": .6 if choice == "BEAR" else .2},
    }}}
    # Normalize so probabilities sum to 1 and the winning choice keeps the max.
    probs = payload["answers"]["market_regime"]["probabilities"]
    total = sum(probs.values())
    payload["answers"]["market_regime"]["probabilities"] = {k: v / total for k, v in probs.items()}
    result = JevResult(status="ok", response=JevResponse.model_validate(payload))
    ctx = build_jev_context(result)
    assert ctx["summary"] == f"{label}｜Confidence 60%"


@pytest.mark.parametrize("status,reason", [
    ("disabled", None),
    ("unavailable", "missing_api_key"),
    ("unavailable", "invalid_api_key"),
    ("unavailable", "timeout"),
    ("unavailable", "network_error"),
    ("unavailable", "authentication_failed"),
    ("invalid_response", "response_validation_failed"),
    ("insufficient_data", "market_state_unavailable"),
])
def test_non_ok_status_is_never_available(status, reason):
    result = JevResult(status=status, reason=reason)
    ctx = build_jev_context(result)
    assert ctx["available"] is False
    assert ctx["summary"] is None
    assert ctx["confidence"] is None
    assert ctx["signals"] is None
    assert ctx["status"] == status
    assert ctx["reason"] == reason


def test_ok_status_with_no_response_is_not_available():
    # Defensive: status=="ok" with no parsed response should never happen in
    # practice (see core.jev_service.evaluate), but the shaping function must
    # not assume it and must not crash if it did.
    result = JevResult(status="ok", response=None)
    ctx = build_jev_context(result)
    assert ctx["available"] is False


def test_context_never_contains_api_key_material(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "super-secret-test-key-never-leak")
    result = JevResult(status="ok", response=JevResponse.model_validate(VALID_RESPONSE))
    ctx = build_jev_context(result)
    assert "super-secret-test-key-never-leak" not in str(ctx)
