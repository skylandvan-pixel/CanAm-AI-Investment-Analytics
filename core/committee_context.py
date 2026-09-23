"""Read-only external decision context surfaced to the Investment Committee UI.

Jev/TypeSafe is a supplementary external decision-intelligence signal, never
an eighth committee member: this module only reshapes an already-evaluated
JevResult (core.jev_models) into the small standardized shape the UI reads.
It never re-parses the TypeSafe response, never re-implements jev_service's
transport/retry/validation logic, and is never included in the AI provider's
prompt, JSON schema, or canonical_fact_packet -- so nothing here can change a
score, Buy/Sell action, or chairman_decision. See docs/JEV_PHASE1.md.
"""
from __future__ import annotations

from core.jev_models import JevResult

_REGIME_LABELS = {"BULL": "偏乐观", "NEUTRAL": "中性", "BEAR": "偏谨慎"}


def build_jev_context(jev_result: JevResult) -> dict:
    """Pure shaping, no I/O. `available` is True only for a fully successful
    call (status == "ok" with a parsed response) -- every other status
    (disabled, unavailable, invalid_response, insufficient_data) is treated
    identically as "not available" so the UI has one simple boolean to check
    and never needs to branch on individual failure reasons. Never includes
    the API key or any transport/header detail -- JevResult itself carries
    neither."""
    available = jev_result.status == "ok" and jev_result.response is not None
    summary = confidence = signals = None
    if available:
        response = jev_result.response
        regime = response.answers.market_regime
        confidence = regime.confidence
        label = _REGIME_LABELS.get(regime.choice, regime.choice)
        summary = f"{label}｜Confidence {confidence:.0%}"
        signals = {
            "market_bullish_probability": response.answers.market_bullish.noul,
            "market_regime_choice": regime.choice,
            "market_regime_probabilities": regime.probabilities.model_dump(),
        }
    return {
        "available": available,
        "status": jev_result.status,
        "summary": summary,
        "confidence": confidence,
        "signals": signals,
        "reason": jev_result.reason,
    }
