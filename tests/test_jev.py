"""Offline Phase 1 tests. Synthetic probabilities are never production data."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from unittest.mock import Mock
import pandas as pd
import pytest
from core import jev_service as service
ORIGINAL_POST = service._post
from core.jev_cli import main
from core.jev_state import build_market_state

VALID = {"model": "jev-test-version", "answers": {
    "market_bullish": {"type": "noul", "noul": .7},
    "market_regime": {"type": "choice", "choice": "BULL", "confidence": .5,
                      "probabilities": {"BULL": .7, "NEUTRAL": .2, "BEAR": .1}}},
    "usage": {"input_tokens": 100, "output_tokens": 10}}
STATE = {"spy": {"close": 100}, "data_timestamp": "2026-09-18"}

@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    from core import jev_cli
    monkeypatch.setattr(jev_cli, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-secret-never-real")
    for name in ("JEV_MODEL", "JEV_TIMEOUT_SECONDS", "JEV_TOTAL_TIMEOUT_SECONDS", "JEV_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(service, "_post", Mock(side_effect=AssertionError("Live transport forbidden")))

def reply(payload=None, status=200, retry_after=None):
    return service.HttpReply(status, json.dumps(VALID if payload is None else payload).encode(), retry_after)

@pytest.mark.parametrize("flag", [None, "false", "FALSE", "0", "invalid"])
def test_disabled_no_key_no_state_no_client(monkeypatch, flag):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    if flag is None:
        monkeypatch.delenv("JEV_ENABLED", raising=False)
    else:
        monkeypatch.setenv("JEV_ENABLED", flag)
    factory, transport = Mock(), Mock()
    assert service.evaluate(factory, transport=transport).status == "disabled"
    factory.assert_not_called(); transport.assert_not_called()

def test_missing_key_before_market(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    factory = Mock()
    assert service.evaluate(factory).reason == "missing_api_key"
    factory.assert_not_called()

def test_valid_contract_and_model_override(monkeypatch):
    monkeypatch.setenv("JEV_MODEL", "jev-pinned")
    transport = Mock(return_value=reply())
    result = service.evaluate(STATE, transport=transport)
    assert result.status == "ok" and result.response.model == "jev-test-version"
    assert result.response.answers.market_bullish.noul == .7
    assert result.response.answers.market_regime.probabilities.BULL == .7
    assert result.regime_change_reason == "insufficient_history"
    payload = json.loads(transport.call_args.args[0])
    assert set(payload) == {"model", "state", "questions"}
    assert payload["model"] == "jev-pinned"
    assert set(payload["questions"]) == {"market_bullish", "market_regime"}
    assert set(payload["questions"]["market_regime"]["criteria"]) == {"BULL", "NEUTRAL", "BEAR"}

@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), "0.7", True])
def test_invalid_noul(value):
    payload = copy.deepcopy(VALID)
    payload["answers"]["market_bullish"]["noul"] = value
    assert service.evaluate(STATE, transport=Mock(return_value=reply(payload))).status == "invalid_response"

@pytest.mark.parametrize("mutation", ["missing_probability", "sum", "wrong_choice", "missing_model", "missing_usage", "missing_answers", "extra", "bad_type"])
def test_bad_response(mutation):
    p = copy.deepcopy(VALID)
    choice = p["answers"]["market_regime"]
    if mutation == "missing_probability": del choice["probabilities"]["BEAR"]
    elif mutation == "sum": choice["probabilities"]["BULL"] = .9
    elif mutation == "wrong_choice": choice["choice"] = "BEAR"
    elif mutation.startswith("missing_"): del p[mutation.removeprefix("missing_")]
    elif mutation == "extra": p["secret"] = "unexpected"
    else: choice["type"] = "noul"
    result = service.evaluate(STATE, transport=Mock(return_value=reply(p)))
    assert result.status == "invalid_response" and result.response is None

@pytest.mark.parametrize("body", [b"not json", b"{}", b"[]", b"x" * 65537])
def test_invalid_json(body):
    assert service.evaluate(STATE, transport=Mock(return_value=service.HttpReply(200, body))).status == "invalid_response"

@pytest.mark.parametrize("exc,reason", [(TimeoutError(), "timeout"), (URLError("offline"), "network_error"), (URLError(TimeoutError()), "timeout"), (RuntimeError("fixture-secret-never-real"), "network_error")])
def test_network_failures(exc, reason, caplog):
    transport = Mock(side_effect=exc)
    result = service.evaluate(STATE, transport=transport)
    assert result.status == "unavailable" and result.reason == reason
    assert result.response is None and transport.call_count == 1
    assert "fixture-secret-never-real" not in result.model_dump_json() + caplog.text

@pytest.mark.parametrize("status", [401, 422, 400, 500, 302])
def test_no_retry_permanent_or_unexpected(status):
    transport = Mock(return_value=reply(status=status))
    assert service.evaluate(STATE, transport=transport).status == "unavailable"
    assert transport.call_count == 1

@pytest.mark.parametrize("status", [429, 529])
@pytest.mark.parametrize("recover", [True, False])
def test_bounded_retry(status, recover):
    transport = Mock(side_effect=[reply(status=status), reply(status=200 if recover else status)])
    sleeper = Mock()
    result = service.evaluate(STATE, transport=transport, sleep=sleeper)
    assert result.status == ("ok" if recover else "unavailable")
    assert transport.call_count == 2 and sleeper.call_count == 1

@pytest.mark.parametrize("retry_after", ["100", "Wed, 23 Sep 2026 10:00:00 GMT", "Infinity"])
def test_retry_deferred(retry_after):
    transport = Mock(return_value=reply(status=429, retry_after=retry_after))
    assert service.evaluate(STATE, transport=transport).reason == "retry_deferred"
    assert transport.call_count == 1

def test_deadline():
    times = iter([0., 0., 13., 13.])
    assert service.evaluate(STATE, transport=Mock(return_value=reply()), clock=lambda: next(times)).reason == "timeout"

@pytest.mark.parametrize("env,value", [("JEV_TIMEOUT_SECONDS", "nan"), ("JEV_MAX_RETRIES", "9"), ("JEV_TOTAL_TIMEOUT_SECONDS", "0")])
def test_invalid_config(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert service.evaluate(Mock()).reason == "invalid_configuration"

def test_insufficient_data():
    assert service.evaluate({}).status == "insufficient_data"
    assert service.evaluate(Mock(side_effect=ValueError())).status == "insufficient_data"

def histories():
    index = pd.bdate_range(end="2026-09-18", periods=300)
    return pd.Series(range(100, 400), index=index, dtype=float), pd.Series([20.], index=pd.to_datetime(["2026-09-18"]))

def test_state_provenance_and_optional_vix():
    spy, vix = histories()
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    state = build_market_state(spy, vix, now=now)
    assert state["benchmark_symbol"] == "SPY"
    assert state["data_timestamp"] == "2026-09-18"
    assert state["generated_timestamp"].startswith("2026-09-21")
    assert state["vix"]["value"] == 20
    assert "VOO" not in json.dumps(state)
    assert build_market_state(spy, pd.Series(dtype=float), now=now)["vix"]["status"] == "unavailable"

def test_today_and_future_excluded():
    spy, vix = histories()
    spy.loc[pd.Timestamp("2026-09-21")] = 99999
    spy.loc[pd.Timestamp("2026-09-22")] = 99999
    state = build_market_state(spy, vix, now=datetime(2026, 9, 21, tzinfo=timezone.utc))
    assert state["spy"]["close"] == 399

@pytest.mark.parametrize("kind", ["short", "stale", "invalid"])
def test_bad_history(kind):
    spy, vix = histories()
    now = datetime(2026, 9, 21, tzinfo=timezone.utc)
    if kind == "short": spy = spy.iloc[-100:]
    if kind == "stale": now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    if kind == "invalid": spy.iloc[-1] = float("inf")
    with pytest.raises(ValueError): build_market_state(spy, vix, now=now)

def test_cli_explicit_only_and_single_attempt(monkeypatch, capsys):
    from core import jev_cli
    factory = Mock(return_value=STATE)
    monkeypatch.setattr(jev_cli, "current_market_state", factory)
    assert main([]) == 0
    factory.assert_not_called()
    transport = Mock(return_value=reply(status=429))
    monkeypatch.setattr(service, "_post", transport)
    assert main(["--live"]) == 1
    assert transport.call_count == 1
    assert "fixture-secret-never-real" not in capsys.readouterr().out

def test_secret_redaction():
    from core.ai import _redact
    assert "fixture-secret-never-real" not in _redact("error fixture-secret-never-real")

def test_existing_analysis_unchanged(mixed_result):
    from core.analytics import canonical_fact_packet
    before = copy.deepcopy(canonical_fact_packet(mixed_result))
    assert service.evaluate(STATE, transport=Mock(side_effect=TimeoutError())).response is None
    assert canonical_fact_packet(mixed_result) == before
    root = Path(__file__).resolve().parents[1]
    # Phase 2 wiring (core.committee_context + app.py's optional jev_context)
    # intentionally gives the UI a read-only Jev hook now, so the old
    # "no 'jev' string anywhere in app.py/ui.py" check no longer applies.
    # What must still hold: Jev never enters the AI provider's canonical
    # fact packet, prompt, or schema -- core/ai.py (where chairman_decision
    # and every score/action are actually produced) stays entirely jev-free.
    assert "jev" not in (root / "core" / "ai.py").read_text().lower()


def test_real_transport_shape_without_network(monkeypatch):
    # Exercise the stdlib adapter itself with an in-memory opener.
    from unittest.mock import MagicMock
    handle = MagicMock()
    handle.__enter__.return_value = handle
    handle.status = 200
    handle.read.return_value = b'{}'
    opener = Mock()
    opener.open.return_value = handle
    monkeypatch.setattr(service, "build_opener", Mock(return_value=opener))
    # Global network guard replaces _post, so inspect the adapter via its
    # original function captured at module import, before fixture execution.
    result = ORIGINAL_POST(b'{}', 'synthetic-transport-key', 2.)
    assert result.status == 200
    request = opener.open.call_args.args[0]
    assert request.full_url == service.ENDPOINT
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer synthetic-transport-key"
    assert opener.open.call_args.kwargs["timeout"] == 2.
    handle.read.assert_called_once_with(service.MAX_RESPONSE_BYTES + 1)


def test_redirect_refused():
    assert service.NoRedirect().redirect_request(None, None, 302, None, None, "https://other.invalid") is None


def test_cli_success_safe_output(monkeypatch, capsys):
    from core import jev_cli
    monkeypatch.setattr(jev_cli, "current_market_state", lambda: STATE)
    monkeypatch.setattr(service, "_post", Mock(return_value=reply()))
    assert main(["--live"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["actual_model_version"] == VALID["model"]
    assert output["market_regime_probabilities"] == VALID["answers"]["market_regime"]["probabilities"]
    assert output["attempts"] == 1


def test_cli_false_probability_and_usage(monkeypatch, capsys):
    from core import jev_cli
    monkeypatch.setattr(jev_cli, "current_market_state", lambda: STATE)
    monkeypatch.setattr(service, "_post", Mock(return_value=reply()))
    assert main(["--live"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["market_bullish_false_probability"] == pytest.approx(.3)
    assert output["false_probability_derivation"] == "1 - API market_bullish.noul"
    assert output["usage"] == VALID["usage"]


def test_cli_disabled_even_with_live_flag(monkeypatch, capsys):
    from core import jev_cli
    monkeypatch.setenv("JEV_ENABLED", "false")
    factory, transport = Mock(), Mock()
    monkeypatch.setattr(jev_cli, "current_market_state", factory)
    monkeypatch.setattr(service, "_post", transport)
    assert main(["--live"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "disabled"
    factory.assert_not_called()
    transport.assert_not_called()


@pytest.mark.parametrize("delay", ["nan", "-1"])
def test_invalid_retry_after_no_retry(delay):
    transport = Mock(return_value=reply(status=429, retry_after=delay))
    assert service.evaluate(STATE, transport=transport).reason == "retry_deferred"
    assert transport.call_count == 1


def test_unexpected_transport_response():
    assert service.evaluate(STATE, transport=Mock(return_value=None)).status == "invalid_response"


@pytest.mark.parametrize("kind,category", [("ssl", "ssl"), ("dns", "dns"), ("socket", "socket"), ("timeout", "timeout"), ("unknown", "unknown")])
def test_safe_diagnostic_classification(kind, category):
    import ssl, socket
    sensitive = "Authorization: Bearer fixture-secret-never-real sensitive-body"
    errors = {"ssl": ssl.SSLCertVerificationError(1, sensitive),
              "dns": socket.gaierror(-2, sensitive), "socket": ConnectionResetError(sensitive),
              "timeout": TimeoutError(sensitive), "unknown": RuntimeError(sensitive)}
    result = service.evaluate(STATE, transport=Mock(side_effect=URLError(errors[kind])), diagnostics=True)
    assert result.diagnostic["category"] == category
    assert result.diagnostic["exception_class"] == "URLError"
    assert "fixture-secret" not in result.model_dump_json()
    assert "Authorization" not in result.model_dump_json()
    assert "sensitive-body" not in result.model_dump_json()


def test_diagnostic_off_by_default():
    result = service.evaluate(STATE, transport=Mock(side_effect=URLError("secret")))
    assert result.diagnostic is None


@pytest.mark.parametrize("status", [401, 422, 429, 529, 302])
def test_diagnostic_http_status(status):
    result = service.evaluate(STATE, transport=Mock(return_value=reply(status=status)), diagnostics=True, max_retries=0)
    assert result.diagnostic["http_status"] == status
    assert result.diagnostic["response_status"] == status


def test_read_failure_retains_status():
    exc = service.ResponseReadFailure(TimeoutError("private body"), 200)
    result = service.evaluate(STATE, transport=Mock(side_effect=exc), diagnostics=True)
    assert result.reason == "timeout"
    assert result.diagnostic["response_status"] == 200
    assert "private body" not in result.model_dump_json()


def test_diagnostics_flag_alone_cannot_call(monkeypatch, capsys):
    from core import jev_cli
    factory = Mock()
    monkeypatch.setattr(jev_cli, "current_market_state", factory)
    assert main(["--diagnostics"]) == 0
    factory.assert_not_called()


@pytest.mark.parametrize("system_key", [None, "system-fixture-key"])
def test_dotenv_key_and_system_precedence(monkeypatch, capsys, caplog, system_key):
    from core import jev_cli
    jev_cli.ENV_FILE.write_text("TYPESAFE_API_KEY=dotenv-fixture-key\nJEV_ENABLED=true\nJEV_MODEL=jev-env\n")
    # Monkeypatch records restoration even when dotenv later populates the key.
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_ENABLED", raising=False)
    if system_key:
        monkeypatch.setenv("TYPESAFE_API_KEY", system_key)
        monkeypatch.setenv("JEV_MODEL", "jev-system")
    transport = Mock(return_value=reply())
    monkeypatch.setattr(service, "_post", transport)
    monkeypatch.setattr(jev_cli, "current_market_state", lambda: STATE)
    assert main(["--live", "--diagnostics"]) == 0
    assert transport.call_args.args[1] == (system_key or "dotenv-fixture-key")
    assert json.loads(transport.call_args.args[0])["model"] == ("jev-system" if system_key else "jev-env")
    output = capsys.readouterr()
    for secret in ("dotenv-fixture-key", "system-fixture-key"):
        assert secret not in output.out + output.err + caplog.text


@pytest.mark.parametrize("system_disabled", [True, False])
def test_dotenv_disabled_no_requests(monkeypatch, capsys, system_disabled):
    from core import jev_cli
    jev_cli.ENV_FILE.write_text("TYPESAFE_API_KEY=dotenv-fixture-key\nJEV_ENABLED=" + ("true" if system_disabled else "false") + "\n")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    if system_disabled:
        monkeypatch.setenv("JEV_ENABLED", "false")
    else:
        monkeypatch.delenv("JEV_ENABLED", raising=False)
    factory, transport = Mock(), Mock()
    monkeypatch.setattr(jev_cli, "current_market_state", factory)
    monkeypatch.setattr(service, "_post", transport)
    assert main(["--live"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "disabled"
    factory.assert_not_called()
    transport.assert_not_called()


def test_dotenv_no_live_does_not_load(monkeypatch):
    from core import jev_cli
    loader = Mock()
    monkeypatch.setattr(jev_cli, "load_dotenv", loader)
    assert main([]) == 0
    loader.assert_not_called()
