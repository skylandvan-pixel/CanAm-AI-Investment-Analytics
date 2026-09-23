"""AppTest-level integration tests for the Jev -> Seven-Member Committee
wiring (Phase 2, wiring only): Jev is a read-only supplementary external
decision context, never an eighth committee member, and must never change
any score, action, or chairman_decision -- see core.committee_context and
docs/JEV_PHASE1.md.

Real TypeSafe network calls are forbidden in pytest -- see
tests/conftest.py::_no_live_jev_network_by_default (autouse, sets
JEV_ENABLED=false and makes core.jev_service._post raise on any use unless a
test explicitly re-patches it, exactly like tests/test_jev.py's own pattern).
"""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import core.ai
import core.jev_state
from core import jev_service
from tests.test_ai import VALID

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
JEV_MARKET_STATE = {"spy": {"close": 100}, "data_timestamp": "2026-09-18"}


def _jev_response(*, noul, choice, probabilities, confidence):
    return {
        "model": "jev-test-version",
        "answers": {
            "market_bullish": {"type": "noul", "noul": noul},
            "market_regime": {"type": "choice", "choice": choice, "confidence": confidence,
                              "probabilities": probabilities},
        },
        "usage": {"input_tokens": 42, "output_tokens": 7},
    }


def _reply(payload, status=200):
    return jev_service.HttpReply(status, json.dumps(payload).encode())


def _app():
    return AppTest.from_file(APP_PATH).run(timeout=20)


def _run_committee(monkeypatch, *, jev_enabled=False, jev_transport=None):
    """Drives the same 'unlock -> open page 3 -> click 启动 AI 投资委员会'
    path every other committee test in tests/test_app.py uses, with
    run_ai_analysis mocked (as a Mock, so its call args can be inspected) and
    the Jev market-state factory redirected away from real yfinance."""
    parsed = core.ai.CommitteeResult.model_validate(VALID)
    mock_run = Mock(return_value=(parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}))
    monkeypatch.setattr(core.ai, "run_ai_analysis", mock_run)
    monkeypatch.setattr(core.jev_state, "current_market_state", lambda: JEV_MARKET_STATE)
    if jev_enabled:
        monkeypatch.setenv("JEV_ENABLED", "true")
        monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-key-never-real")
    if jev_transport is not None:
        monkeypatch.setattr(jev_service, "_post", jev_transport)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception
    return app, parsed, mock_run


def _rendered_text(app):
    return "\n".join(m.value for m in app.markdown) + "\n" + "\n".join(c.value for c in app.caption)


def _assert_committee_untouched_by_jev(app, parsed, mock_run):
    """The invariant every scenario below must hold: run_ai_analysis is
    called with exactly the portfolio result and nothing else (no Jev data
    smuggled in as an extra arg/kwarg), and the rendered chairman_decision /
    action plan match the mocked baseline exactly."""
    assert mock_run.call_count == 1
    assert len(mock_run.call_args.args) == 1
    assert mock_run.call_args.kwargs == {}
    assert app.session_state["ai_result"].chairman_decision == parsed.chairman_decision
    assert app.session_state["ai_result"].action_plan == parsed.action_plan
    assert app.session_state["ai_result"].members == parsed.members
    assert parsed.chairman_decision in _rendered_text(app)


def test_a_jev_disabled_committee_output_unchanged(monkeypatch):
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=False)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    ctx = app.session_state["jev_context"]
    assert ctx["available"] is False
    assert ctx["status"] == "disabled"
    assert "Jev 外部决策参考" not in _rendered_text(app)


@pytest.mark.parametrize("exc,expected_reason", [
    (TimeoutError(), "timeout"),
    (RuntimeError("offline"), "network_error"),
])
def test_b_jev_unavailable_committee_completes_normally(monkeypatch, exc, expected_reason):
    transport = Mock(side_effect=exc)
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=True, jev_transport=transport)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    ctx = app.session_state["jev_context"]
    assert ctx["available"] is False
    assert ctx["status"] == "unavailable"
    assert ctx["reason"] == expected_reason
    assert "Jev 外部决策参考" not in _rendered_text(app)


def test_c_jev_ok_context_enters_committee_and_ui(monkeypatch):
    payload = _jev_response(noul=.6, choice="NEUTRAL", probabilities={"BULL": .3, "NEUTRAL": .5, "BEAR": .2}, confidence=.72)
    transport = Mock(return_value=_reply(payload))
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=True, jev_transport=transport)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    ctx = app.session_state["jev_context"]
    assert ctx["available"] is True
    assert ctx["status"] == "ok"
    assert ctx["summary"] == "中性｜Confidence 72%"
    assert "Jev 外部决策参考：中性｜Confidence 72%" in _rendered_text(app)


def test_d_jev_strong_buy_does_not_change_chairman_or_scores(monkeypatch):
    payload = _jev_response(noul=.97, choice="BULL", probabilities={"BULL": .93, "NEUTRAL": .05, "BEAR": .02}, confidence=.93)
    transport = Mock(return_value=_reply(payload))
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=True, jev_transport=transport)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    ctx = app.session_state["jev_context"]
    assert ctx["available"] is True
    assert ctx["summary"] == "偏乐观｜Confidence 93%"


def test_e_jev_strong_sell_does_not_change_chairman_or_scores(monkeypatch):
    payload = _jev_response(noul=.03, choice="BEAR", probabilities={"BULL": .02, "NEUTRAL": .05, "BEAR": .93}, confidence=.93)
    transport = Mock(return_value=_reply(payload))
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=True, jev_transport=transport)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    ctx = app.session_state["jev_context"]
    assert ctx["available"] is True
    assert ctx["summary"] == "偏谨慎｜Confidence 93%"


@pytest.mark.parametrize("transport,expected_status", [
    (Mock(return_value=jev_service.HttpReply(200, b"not json")), "invalid_response"),
    (Mock(side_effect=TimeoutError()), "unavailable"),
])
def test_f_jev_malformed_or_timeout_report_still_renders(monkeypatch, transport, expected_status):
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=True, jev_transport=transport)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    assert app.session_state["jev_context"]["status"] == expected_status
    assert "Jev 外部决策参考" not in _rendered_text(app)


def test_g_api_key_never_leaks_into_context_ui_or_logs(monkeypatch, caplog):
    transport = Mock(side_effect=RuntimeError("boom fixture-key-never-real"))
    app, parsed, mock_run = _run_committee(monkeypatch, jev_enabled=True, jev_transport=transport)
    _assert_committee_untouched_by_jev(app, parsed, mock_run)
    secret = "fixture-key-never-real"
    assert secret not in str(app.session_state["jev_context"])
    assert secret not in _rendered_text(app)
    assert secret not in caplog.text
