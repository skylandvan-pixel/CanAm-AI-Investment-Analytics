from __future__ import annotations

import json
import logging

import pytest

from core.ai import CommitteeResult, ProviderUnavailable, _redact, portfolio_fingerprint, run_ai_analysis
from core.auth import validate_beta_code


VALID = {
    "members": [
        {"role": role, "stance": "持有并优化", "conclusion": f"{role} conclusion"}
        for role in ["macro", "portfolio", "risk", "valuation_data", "tax", "action_rebalancing"]
    ],
    "majority_view": "维持核心配置并优化结构。",
    "main_concern": "集中度需要持续观察。",
    "chairman_decision": "保持纪律，以条件触发方式调整。",
    "actions": [
        {"horizon":"now", "trigger":"若集中度维持高位", "action":"审视新增资金方向", "reason":"避免进一步放大风险"},
        {"horizon":"one_month", "trigger":"若风险指标继续恶化", "action":"考虑分阶段降低集中", "reason":"控制单一来源风险"},
        {"horizon":"three_month", "trigger":"若结构偏离目标仍未改善", "action":"进行组合级再平衡复核", "reason":"恢复配置纪律"},
    ],
}


class FakeProvider:
    name = "fake"
    model = "test"
    def __init__(self, payload=VALID):
        self.calls = 0
        self.payload = payload
    def generate(self, prompt, schema):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


def test_promotion_code_validation():
    assert validate_beta_code("OPEN-SESAME", ("OPEN-SESAME",))
    assert not validate_beta_code("wrong", ("OPEN-SESAME",))


def test_locked_state_requires_explicit_validation():
    assert not validate_beta_code("", ("OPEN-SESAME",))


def test_ai_is_explicit_trigger_only(mixed_result):
    provider = FakeProvider()
    portfolio_fingerprint(mixed_result)
    assert provider.calls == 0


def test_ai_cache_avoids_duplicate_call(tmp_path, mixed_result):
    provider = FakeProvider()
    first, meta1 = run_ai_analysis(mixed_result, provider=provider, cache_dir=tmp_path)
    second, meta2 = run_ai_analysis(mixed_result, provider=provider, cache_dir=tmp_path)
    assert first == second
    assert provider.calls == 1
    assert not meta1["cache_hit"] and meta2["cache_hit"]


def test_fingerprint_is_stable(mixed_result):
    assert portfolio_fingerprint(mixed_result) == portfolio_fingerprint(mixed_result)


def test_structured_output_validation_fails_closed(tmp_path, mixed_result):
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider({"made_up": "facts"}), cache_dir=tmp_path)


def test_ai_cannot_mutate_deterministic_facts(tmp_path, mixed_result):
    before = (mixed_result.portfolio_score, mixed_result.top3_concentration)
    run_ai_analysis(mixed_result, provider=FakeProvider(), cache_dir=tmp_path)
    assert (mixed_result.portfolio_score, mixed_result.top3_concentration) == before


def test_six_member_role_contract_rejects_duplicates(tmp_path, mixed_result):
    broken = dict(VALID)
    broken["members"] = [VALID["members"][0]] * 6
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_api_unavailable_without_key(monkeypatch, mixed_result, tmp_path):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)


class _RaisingProvider:
    name = "fake"
    model = "test-model"

    def __init__(self, exc):
        self._exc = exc

    def generate(self, prompt, schema):
        raise self._exc


def test_generate_content_failure_is_logged_with_stage(caplog, mixed_result, tmp_path):
    caplog.set_level(logging.ERROR, logger="canam.ai")
    provider = _RaisingProvider(RuntimeError("simulated: model not found"))
    with pytest.raises(RuntimeError):
        run_ai_analysis(mixed_result, provider=provider, cache_dir=tmp_path)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "stage=generate_content" in message
    assert "provider=fake" in message
    assert "model=test-model" in message
    assert "exception=RuntimeError" in message
    assert "simulated: model not found" in message


def test_provider_init_failure_is_logged(caplog, monkeypatch, mixed_result, tmp_path):
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "not-a-real-provider")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert len(caplog.records) == 1
    assert "stage=provider_init" in caplog.records[0].getMessage()


def test_response_parse_failure_is_logged(caplog, mixed_result, tmp_path):
    caplog.set_level(logging.ERROR, logger="canam.ai")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider({"made_up": "facts"}), cache_dir=tmp_path)
    assert len(caplog.records) == 1
    assert "stage=response_parse" in caplog.records[0].getMessage()


def test_successful_run_emits_no_failure_log(caplog, mixed_result, tmp_path):
    caplog.set_level(logging.ERROR, logger="canam.ai")
    run_ai_analysis(mixed_result, provider=FakeProvider(), cache_dir=tmp_path)
    assert caplog.records == []


def test_ai_failure_log_never_contains_full_fact_packet(caplog, mixed_result, tmp_path):
    """The diagnostic log must never become a portfolio-data dump: only the
    exception, provider/model/stage, and boolean key-presence flags."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    provider = _RaisingProvider(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        run_ai_analysis(mixed_result, provider=provider, cache_dir=tmp_path)
    message = caplog.records[0].getMessage()
    assert str(round(mixed_result.portfolio_score)) not in message or "portfolio_score" not in message
    assert "top_direct" not in message and "true_exposures" not in message


def test_secrets_are_redacted_from_ai_failure_logs(caplog, monkeypatch, mixed_result, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFAKESECRETVALUE1234567890")
    monkeypatch.setenv("CANAM_BETA_CODES", "SUPER-SECRET-BETA-CODE")
    caplog.set_level(logging.ERROR, logger="canam.ai")
    provider = _RaisingProvider(RuntimeError(
        "request failed: Authorization: Bearer AIzaSyFAKESECRETVALUE1234567890, "
        "beta code SUPER-SECRET-BETA-CODE was in the payload"
    ))
    with pytest.raises(RuntimeError):
        run_ai_analysis(mixed_result, provider=provider, cache_dir=tmp_path)
    message = caplog.records[0].getMessage()
    assert "AIzaSyFAKESECRETVALUE1234567890" not in message
    assert "SUPER-SECRET-BETA-CODE" not in message
    assert "Bearer" not in message
    assert "[REDACTED]" in message
    # Presence is reported as a boolean, never the value.
    assert "gemini_key_present=True" in message


def test_redact_strips_generic_credential_shapes():
    assert _redact("token AIzaSyABCDEFGHIJKLMNOPQRSTUV1234567") == "token [REDACTED]"
    assert _redact("key sk-ant-abcdefghijklmnopqrstuvwxyz123456") == "key [REDACTED]"
    assert _redact("Authorization: Bearer abcdefghijklmnop1234567890") == "[REDACTED]"
    assert _redact("") == ""
    assert _redact("no secrets here") == "no secrets here"


# --- Gemini structured-output regression (response_json_schema fix) ------

def _schema_contains_additional_properties(schema) -> bool:
    """CommitteeResult uses ConfigDict(extra="forbid"), which makes Pydantic
    emit `additionalProperties` in its JSON Schema -- this is exactly the
    keyword the production error named as unsupported by `response_schema`."""
    if isinstance(schema, dict):
        if "additionalProperties" in schema:
            return True
        return any(_schema_contains_additional_properties(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_schema_contains_additional_properties(v) for v in schema)
    return False


def test_committee_schema_contains_additional_properties():
    """Sanity check that the exact production failure condition is real:
    the generated schema does contain additionalProperties, so the fix must
    route it through a field that supports that JSON Schema keyword."""
    schema = CommitteeResult.model_json_schema()
    assert _schema_contains_additional_properties(schema)


class _FakeGeminiModels:
    """Records the exact call shape GeminiProvider sends to the real SDK's
    generate_content, without making any network call."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        class _Response:
            text = self.response_text
        return _Response()


class _FakeGeminiClient:
    def __init__(self, api_key, *, response_text):
        self.models = _FakeGeminiModels(response_text)


def _install_fake_gemini_client(monkeypatch, response_text):
    import google.genai as genai
    fake_client_holder = {}

    def _factory(api_key):
        client = _FakeGeminiClient(api_key, response_text=response_text)
        fake_client_holder["client"] = client
        return client

    monkeypatch.setattr(genai, "Client", _factory)
    return fake_client_holder


class _FlakyGeminiModels:
    """Consumes one `outcome` per generate_content call: an Exception
    instance is raised, anything else is returned as `response.text`."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        class _Response:
            text = outcome
        return _Response()


def _install_flaky_gemini_client(monkeypatch, outcomes):
    import google.genai as genai
    fake_client_holder = {}

    def _factory(api_key):
        class _Client:
            def __init__(self):
                self.models = _FlakyGeminiModels(outcomes)
        client = _Client()
        fake_client_holder["client"] = client
        return client

    monkeypatch.setattr(genai, "Client", _factory)
    return fake_client_holder


def _server_error(status="UNAVAILABLE", code=503, message="high demand"):
    from google.genai import errors
    return errors.ServerError(code, {"error": {"code": code, "message": message, "status": status}})


def _client_error(status="INVALID_ARGUMENT", code=400, message="invalid argument"):
    from google.genai import errors
    return errors.ClientError(code, {"error": {"code": code, "message": message, "status": status}})


def test_gemini_provider_uses_response_json_schema_not_response_schema(monkeypatch):
    """A/C: the request must be built with response_json_schema carrying the
    full CommitteeResult JSON schema, and response_schema must NOT be set
    at the same time (the SDK requires exactly one of the two)."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    holder = _install_fake_gemini_client(monkeypatch, response_text=json.dumps(VALID, ensure_ascii=False))

    provider = GeminiProvider()
    schema = CommitteeResult.model_json_schema()
    text = provider.generate("prompt text", schema)

    assert text == json.dumps(VALID, ensure_ascii=False)
    call = holder["client"].models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert config.response_schema is None
    # Pre-Gemini-3 models keep the legacy sampling/thinking-budget shape.
    assert config.temperature == 0
    assert config.thinking_config.thinking_budget == 0
    assert config.thinking_config.thinking_level is None


def test_gemini_3_provider_uses_thinking_level_and_drops_temperature(monkeypatch):
    """Regression test for the production 400 INVALID_ARGUMENT: gemini-3.x
    rejects temperature and thinking_budget, so gemini-3.6-flash (and any
    other gemini-3* model) must be sent without temperature, without
    thinking_budget, and with thinking_level="minimal" instead -- while still
    preserving response_json_schema/response_mime_type and never falling
    back to the legacy response_schema field."""
    from google.genai import types
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    holder = _install_fake_gemini_client(monkeypatch, response_text=json.dumps(VALID, ensure_ascii=False))

    provider = GeminiProvider()
    schema = CommitteeResult.model_json_schema()
    text = provider.generate("prompt text", schema)

    assert text == json.dumps(VALID, ensure_ascii=False)
    call = holder["client"].models.calls[0]
    assert call["model"] == "gemini-3.6-flash"
    config = call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert config.response_schema is None
    assert config.temperature is None
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_run_ai_analysis_end_to_end_with_gemini_3_provider_succeeds(monkeypatch, caplog, mixed_result, tmp_path):
    """A + B + C: the full run_ai_analysis path also works end-to-end against
    the gemini-3.6-flash request shape, with a mocked SDK call (no network)
    and no diagnostic failure logged on success."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _install_fake_gemini_client(monkeypatch, response_text=json.dumps(VALID, ensure_ascii=False))

    result, meta = run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert result.chairman_decision == VALID["chairman_decision"]
    assert meta["provider"] == "gemini" and meta["model"] == "gemini-3.6-flash"
    assert caplog.records == []


def test_run_ai_analysis_with_gemini_3_provider_still_fails_closed_on_bad_json(monkeypatch, caplog, mixed_result, tmp_path):
    """D + E + F: a schema-violating gemini-3.6-flash response must still
    raise ProviderUnavailable and still be diagnosed at stage=response_parse."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _install_fake_gemini_client(monkeypatch, response_text=json.dumps({"made_up": "facts"}))

    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert len(caplog.records) == 1
    assert "stage=response_parse" in caplog.records[0].getMessage()
    assert "provider=gemini" in caplog.records[0].getMessage()


def test_gemini_provider_default_model_is_gemini_3_6_flash(monkeypatch):
    """G: the canonical default (used when GEMINI_MODEL is unset) must match
    the current production target, not a stale placeholder."""
    from core.ai import GeminiProvider

    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    _install_fake_gemini_client(monkeypatch, response_text="{}")

    provider = GeminiProvider()
    assert provider.model == "gemini-3.6-flash"


def _no_sleep_calls(monkeypatch):
    """Mocks time.sleep so retry-backoff tests run instantly and deterministically."""
    calls = []
    monkeypatch.setattr("core.ai.time.sleep", lambda seconds: calls.append(seconds))
    return calls


def test_gemini_provider_succeeds_first_attempt_no_retry_no_sleep(monkeypatch):
    """A: the golden path is untouched -- one call, no sleep, when the first
    attempt already succeeds."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(monkeypatch, outcomes=[json.dumps(VALID, ensure_ascii=False)])

    provider = GeminiProvider()
    text = provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert text == json.dumps(VALID, ensure_ascii=False)
    assert len(holder["client"].models.calls) == 1
    assert sleeps == []


def test_gemini_provider_retries_once_on_503_then_succeeds(monkeypatch):
    """B: one transient 503, then success -- exactly two calls, exactly one
    backoff sleep (~2s), and the result still parses normally."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), json.dumps(VALID, ensure_ascii=False)],
    )

    provider = GeminiProvider()
    text = provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert text == json.dumps(VALID, ensure_ascii=False)
    assert len(holder["client"].models.calls) == 2
    assert sleeps == [2]
    assert CommitteeResult.model_validate_json(text).chairman_decision == VALID["chairman_decision"]


def test_gemini_provider_retries_twice_on_503_then_succeeds(monkeypatch):
    """C: two transient 503s, then success -- exactly three calls, exactly
    two backoff sleeps (~2s, ~5s), and the result still parses normally."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), json.dumps(VALID, ensure_ascii=False)],
    )

    provider = GeminiProvider()
    text = provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert text == json.dumps(VALID, ensure_ascii=False)
    assert len(holder["client"].models.calls) == 3
    assert sleeps == [2, 5]
    assert CommitteeResult.model_validate_json(text).chairman_decision == VALID["chairman_decision"]


def test_gemini_provider_fails_closed_after_three_503s(monkeypatch, caplog):
    """D: 503 on every one of the 3 attempts -- exactly three calls, exactly
    two sleeps (none after the final attempt), fail closed by re-raising the
    ServerError, and the existing final diagnostic logging still fires
    (via run_ai_analysis's stage=generate_content handler)."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 3
    assert sleeps == [2, 5]


def test_run_ai_analysis_fails_closed_after_three_503s_logs_final_failure(monkeypatch, caplog, mixed_result, tmp_path):
    """D (full path): after exhausting retries, run_ai_analysis's existing
    stage=generate_content diagnostic logging must still fire exactly once,
    with no leaked secrets."""
    caplog.set_level(logging.WARNING, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error()],
    )

    with pytest.raises(Exception):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    messages = [r.getMessage() for r in caplog.records]
    retry_messages = [m for m in messages if "transient failure" in m]
    final_messages = [m for m in messages if m.startswith("AI provider failure:")]
    assert len(retry_messages) == 2
    assert "attempt=1/3" in retry_messages[0]
    assert "attempt=2/3" in retry_messages[1]
    assert len(final_messages) == 1
    assert "stage=generate_content" in final_messages[0]
    assert "test-key-not-real" not in "\n".join(messages)


def test_gemini_provider_does_not_retry_on_400_invalid_argument(monkeypatch):
    """E: a permanent 400 INVALID_ARGUMENT must fail closed immediately --
    exactly one call, zero retries, zero sleeps."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error()])

    provider = GeminiProvider()
    with pytest.raises(errors.ClientError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 1
    assert sleeps == []


def test_gemini_provider_does_not_retry_on_401_auth_error(monkeypatch):
    """F: an authentication/permanent error must not be retried either --
    exactly one call, no retry."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_client_error(status="UNAUTHENTICATED", code=401, message="invalid API key")],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ClientError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 1
    assert sleeps == []


def test_gemini_provider_does_not_retry_on_403_permission_denied(monkeypatch):
    """F: 403 permission-denied is likewise a permanent error, not retried."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_client_error(status="PERMISSION_DENIED", code=403, message="forbidden")],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ClientError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 1
    assert sleeps == []


def test_gemini_provider_invalid_json_after_retry_still_fails_closed(monkeypatch, caplog, mixed_result, tmp_path):
    """G + H: a retry that eventually returns a schema-invalid JSON payload
    must still fail closed at response_parse (retries only apply to the
    transport-level 503, never to a Pydantic validation failure)."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), json.dumps({"made_up": "facts"})],
    )

    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert len(caplog.records) == 1
    assert "stage=response_parse" in caplog.records[0].getMessage()


def test_run_ai_analysis_end_to_end_with_gemini_provider_succeeds(monkeypatch, caplog, mixed_result, tmp_path):
    """D + F: a valid mocked Gemini response parses to CommitteeResult through
    the full run_ai_analysis path (configured_provider -> GeminiProvider ->
    generate -> validation), and no failure is logged on success."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    _install_fake_gemini_client(monkeypatch, response_text=json.dumps(VALID, ensure_ascii=False))

    result, meta = run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert result.chairman_decision == VALID["chairman_decision"]
    assert meta["provider"] == "gemini" and meta["model"] == "gemini-2.5-flash"
    assert caplog.records == []


def test_run_ai_analysis_with_gemini_provider_still_fails_closed_on_bad_json(monkeypatch, caplog, mixed_result, tmp_path):
    """E + F: a schema-violating Gemini response must still raise
    ProviderUnavailable (fail closed, validation not weakened), and the
    diagnostic logger must still capture it at stage=response_parse."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    _install_fake_gemini_client(monkeypatch, response_text=json.dumps({"made_up": "facts"}))

    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert len(caplog.records) == 1
    assert "stage=response_parse" in caplog.records[0].getMessage()
    assert "provider=gemini" in caplog.records[0].getMessage()
