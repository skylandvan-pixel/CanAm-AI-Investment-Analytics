from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from core.ai import CommitteeResult, ProviderUnavailable, _redact, portfolio_fingerprint, run_ai_analysis
from core.auth import validate_beta_code


# Tickers here are exactly what tests/conftest.py::mixed_result produces
# (holdings NVDA/VOO/SGOV; VOO's look-through adds AAPL/MSFT/AMZN/META/
# GOOGL/GOOG/AVGO) -- run_ai_analysis validates every Action Plan ticker
# reference against the real fact packet, so any test that exercises the
# full run_ai_analysis path (not just GeminiProvider.generate in isolation)
# needs VALID's tickers to actually be present in that portfolio.
VALID = {
    "members": [
        {"role": role, "stance": "持有并优化", "conclusion": f"{role} conclusion"}
        for role in ["macro", "portfolio", "risk", "valuation_data", "tax", "action_rebalancing"]
    ],
    "majority_view": "维持核心配置并优化结构。",
    "main_concern": "集中度需要持续观察。",
    "chairman_decision": "保持纪律，以条件触发方式调整。",
    "action_plan": {
        "strategy_now": "防守优先于进攻，控制单一集中度。",
        "top_actions": ["控制NVDA集中度", "提高核心宽基比例"],
        "do_now": ["复核NVDA仓位规模", "关注即将公布的财报"],
        "do_not_now": ["不要因单日波动一次性清仓", "不要为分散而分散新增低信念持仓"],
        "security_actions": [
            {
                "ticker": "NVDA", "action": "REDUCE_ON_REBOUND", "priority": "高",
                "target": "反弹时分阶段减仓约15%-20%", "trigger": "若反弹至近期压力区",
                "reason": "单一标的集中度过高", "position_reduction_pct": 15.0,
            },
        ],
        "timeline_now": ["复核NVDA仓位"],
        "timeline_30_days": ["关注NVDA财报"],
        "timeline_3_months": ["分批降低NVDA权重"],
        "timeline_6_12_months": ["提高VOO等核心宽基占比"],
        "reassessment_triggers": [
            {"event_or_condition": "NVDA 下一次财报后", "affected_holdings": ["NVDA"], "reassess": "重新评估集中度与 True Exposure"},
        ],
        "checklist": ["复核NVDA仓位", "关注财报", "季度复核集中度"],
    },
}


# CommitteeResult's serialized JSON Schema, as of the pre-Step-1 baseline
# (commit 6217d51) -- the Action Plan reduction pass is prompt/validation
# only and must not grow this at all (see test_schema_bytes_unchanged_by_
# reduction_pass and the Step 1 report). Step 2A made one deliberate,
# approved, flat, optional-field exception (SecurityAction.position_
# reduction_pct, for the local Trade Impact Preview -- see core.trade_preview
# and the Step 2A report) and rebaselined these two constants accordingly;
# any *further* growth beyond that should again be treated as a real finding.
# Rebaselined again after the Codex pre-commit fix that dropped gt/lt from
# position_reduction_pct (see test_schema_bytes_match_step_2a_approved_baseline).
_BASELINE_SCHEMA_BYTES = 4103
_BASELINE_SCHEMA_DEFS = 4


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


# ---------------------------------------------------------------------------
# Action Plan upgrade: schema shape + fail-closed guardrails.
# ---------------------------------------------------------------------------

def _action_plan(**overrides):
    plan = json.loads(json.dumps(VALID["action_plan"]))
    plan.update(overrides)
    return plan


def test_action_plan_valid_payload_parses(mixed_result):
    """1: the full simplified five-section Action Plan shape validates
    cleanly against the real fact packet for a portfolio that actually holds
    every referenced ticker."""
    from core.ai import _validate_action_plan_facts, _validate_action_plan_limits
    from core.analytics import canonical_fact_packet

    parsed = CommitteeResult.model_validate(VALID)
    _validate_action_plan_facts(parsed.action_plan, canonical_fact_packet(mixed_result))
    _validate_action_plan_limits(parsed.action_plan)
    assert parsed.action_plan.strategy_now == VALID["action_plan"]["strategy_now"]


def test_legacy_flat_actions_schema_no_longer_accepted():
    """Regression guard: the old shallow now/one_month/three_month IF/THEN
    list must not silently pass as a valid Action Plan any more."""
    legacy = dict(VALID)
    legacy["actions"] = [{"horizon": "now", "trigger": "t", "action": "a", "reason": "r"}]
    del legacy["action_plan"]
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate(legacy)


def test_legacy_eleven_section_schema_no_longer_accepted():
    """Regression guard: the old 11-section Action Plan (top_changes,
    target_structure, target_migration, scenarios, decision_calendar,
    no_change, nested timeline) must not silently pass as valid any more --
    the simplified five-section shape is required instead."""
    legacy_plan = {
        "strategy_now": "s", "top_changes": ["a"], "do_now": ["a", "b"], "do_not_now": ["a", "b"],
        "security_actions": [{"ticker": "NVDA", "action": "持有", "priority": "高", "reason": "r"}],
        "target_structure": {"current": "c", "next_3_months": "c", "next_6_12_months": "c"},
        "timeline": {
            "now": {"objective": "o", "actions": ["a"], "review_trigger": "t"},
            "next_30_days": {"objective": "o", "actions": ["a"], "review_trigger": "t"},
            "next_3_months": {"objective": "o", "actions": ["a"], "review_trigger": "t"},
            "next_6_12_months": {"objective": "o", "actions": ["a"], "review_trigger": "t"},
        },
        "checklist": ["a", "b", "c", "d", "e"],
    }
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate({**VALID, "action_plan": legacy_plan})


@pytest.mark.parametrize("field", [
    "strategy_now", "top_actions", "do_now", "do_not_now", "security_actions",
    "timeline_now", "timeline_30_days", "timeline_3_months", "timeline_6_12_months",
    "reassessment_triggers", "checklist",
])
def test_action_plan_required_fields_are_enforced(field):
    """Every one of the five sections' fields is required -- a response
    missing any one of them must fail validation (fail closed), not
    silently render a blank section."""
    broken = _action_plan()
    del broken[field]
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate({**VALID, "action_plan": broken})


@pytest.mark.parametrize("field", [
    "strategy_now", "top_actions", "do_now", "do_not_now", "security_actions",
    "timeline_now", "timeline_30_days", "timeline_3_months", "timeline_6_12_months",
    "reassessment_triggers", "checklist",
])
def test_action_plan_all_five_sections_are_representable(field):
    """1 (test-list): each of the five conceptual sections -- current action
    summary (strategy_now/top_actions/do_now/do_not_now), key security
    actions, action timeline (4 horizons), reassessment triggers, and the
    execution checklist -- is representable in the schema and present on a
    valid parsed plan."""
    parsed = CommitteeResult.model_validate(VALID)
    assert getattr(parsed.action_plan, field)


def test_action_plan_empty_do_now_rejected():
    """do_now must be non-empty -- a required field cannot be an empty list."""
    broken = _action_plan(do_now=[])
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate({**VALID, "action_plan": broken})


def test_action_plan_top_actions_capped_at_three_by_local_validation():
    """2: top_actions cannot exceed 3 after local validation -- the schema
    itself accepts more (kept flat/unbounded for the provider), but
    _validate_action_plan_limits fails closed on the business rule."""
    from core.ai import _validate_action_plan_limits

    too_many = _action_plan(top_actions=["改动一", "改动二", "改动三", "改动四"])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": too_many})
    with pytest.raises(ValueError, match="top_actions"):
        _validate_action_plan_limits(parsed.action_plan)


def test_action_plan_security_actions_capped_at_five_by_local_validation():
    """3: security_actions cannot exceed 5 after local validation."""
    from core.ai import _validate_action_plan_limits

    base = VALID["action_plan"]["security_actions"][0]
    too_many = _action_plan(security_actions=[{**base, "ticker": t} for t in ["NVDA", "VOO", "SGOV", "AAPL", "MSFT", "AMZN"]])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": too_many})
    with pytest.raises(ValueError, match="security_actions"):
        _validate_action_plan_limits(parsed.action_plan)


def test_action_plan_reassessment_triggers_capped_at_three_by_local_validation():
    """4: reassessment_triggers cannot exceed 3 after local validation."""
    from core.ai import _validate_action_plan_limits

    base = VALID["action_plan"]["reassessment_triggers"][0]
    too_many = _action_plan(reassessment_triggers=[dict(base) for _ in range(4)])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": too_many})
    with pytest.raises(ValueError, match="reassessment_triggers"):
        _validate_action_plan_limits(parsed.action_plan)


def test_action_plan_checklist_capped_at_six_by_local_validation():
    """5: checklist cannot exceed 6 after local validation."""
    from core.ai import _validate_action_plan_limits

    too_many = _action_plan(checklist=[f"任务{i}" for i in range(7)])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": too_many})
    with pytest.raises(ValueError, match="checklist"):
        _validate_action_plan_limits(parsed.action_plan)


@pytest.mark.parametrize("horizon", ["timeline_now", "timeline_30_days", "timeline_3_months", "timeline_6_12_months"])
def test_action_plan_timeline_horizon_capped_at_three_by_local_validation(horizon):
    """6: each timeline horizon cannot exceed 3 items after local
    validation."""
    from core.ai import _validate_action_plan_limits

    too_many = _action_plan(**{horizon: ["a", "b", "c", "d"]})
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": too_many})
    with pytest.raises(ValueError, match=horizon):
        _validate_action_plan_limits(parsed.action_plan)


def test_action_plan_within_limits_passes_local_validation():
    """Sanity counterpart: a plan at or under every cap must not raise."""
    from core.ai import _validate_action_plan_limits

    parsed = CommitteeResult.model_validate(VALID)
    _validate_action_plan_limits(parsed.action_plan)  # must not raise


def test_action_plan_rejects_vague_action_label_outside_enum():
    """Action must be one of the concrete, constrained labels -- an
    unconstrained/vague free-text action is rejected at the schema level."""
    broken = _action_plan()
    broken["security_actions"][0]["action"] = "适度优化"
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate({**VALID, "action_plan": broken})


def test_action_plan_unknown_ticker_in_security_actions_fails_closed(mixed_result):
    """7: a security action referencing a ticker that is not part of this
    portfolio's canonical fact packet must fail closed, never render."""
    from core.ai import _validate_action_plan_facts
    from core.analytics import canonical_fact_packet

    broken = _action_plan()
    broken["security_actions"][0]["ticker"] = "TSLA"
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="unverified ticker"):
        _validate_action_plan_facts(parsed.action_plan, canonical_fact_packet(mixed_result))


def test_action_plan_unknown_ticker_in_reassessment_trigger_fails_closed(mixed_result):
    """8: an unverified ticker in a reassessment trigger's affected_holdings
    must also fail closed."""
    from core.ai import _validate_action_plan_facts
    from core.analytics import canonical_fact_packet

    broken = _action_plan()
    broken["reassessment_triggers"][0]["affected_holdings"] = ["TSLA"]
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="unverified ticker"):
        _validate_action_plan_facts(parsed.action_plan, canonical_fact_packet(mixed_result))


def test_reassessment_trigger_has_no_date_field():
    """9: no event date field is required -- ReassessmentTrigger's schema
    exposes only event_or_condition/affected_holdings/reassess, and a
    plan validates fine with a purely descriptive condition and no date of
    any kind."""
    schema = CommitteeResult.model_json_schema()
    trigger_props = set(schema["$defs"]["ReassessmentTrigger"]["properties"])
    assert trigger_props == {"event_or_condition", "affected_holdings", "reassess"}
    assert "date" not in trigger_props

    parsed = CommitteeResult.model_validate(VALID)
    assert parsed.action_plan.reassessment_triggers[0].event_or_condition == "NVDA 下一次财报后"


def test_reassessment_trigger_rejects_unexpected_date_field():
    """Fail-closed companion to the above: extra="forbid" means even if the
    model tried to smuggle a date field onto a trigger, it is rejected."""
    broken = _action_plan()
    broken["reassessment_triggers"][0]["date"] = "2026-11-19"
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate({**VALID, "action_plan": broken})


def test_action_plan_schema_has_no_scenario_probability_field():
    """10: no scenario probability field is required or accepted -- the old
    scenarios/weight_label module is gone entirely, so the schema must not
    expose any field name suggesting a statistical probability or weight."""
    schema_text = json.dumps(CommitteeResult.model_json_schema(), ensure_ascii=False).lower()
    for forbidden in ("probability", "weight_label", "scenario"):
        assert forbidden not in schema_text


def test_action_plan_schema_has_no_share_count_field():
    """The app never sends per-share quantity/price to the AI provider (see
    test_fact_packet_is_aggregated_not_raw_holdings in
    tests/test_analytics.py), so the Action Plan schema must not expose any
    field that could hold an invented share count."""
    schema_text = json.dumps(CommitteeResult.model_json_schema(), ensure_ascii=False).lower()
    for forbidden in ("share_count", "shares", "quantity", "num_shares"):
        assert forbidden not in schema_text


def test_action_plan_old_standalone_modules_are_gone():
    """Section 3: the old independent Top Changes / Target Migration /
    Target Structure / Scenario Analysis / Decision Calendar / No Change
    modules must no longer exist as separate schema fields."""
    schema = CommitteeResult.model_json_schema()
    action_plan_props = set(schema["$defs"]["ActionPlan"]["properties"])
    for removed in (
        "top_changes", "target_migration", "target_structure",
        "scenarios", "decision_calendar", "no_change", "timeline",
    ):
        assert removed not in action_plan_props


def test_run_ai_analysis_fails_closed_when_action_plan_references_unknown_ticker(tmp_path, mixed_result):
    """Full path (not just the schema layer): run_ai_analysis must fail
    closed -- via ProviderUnavailable, exactly like any other response_parse
    failure -- when the Action Plan references a ticker outside this
    portfolio."""
    broken = _action_plan()
    broken["security_actions"][0]["ticker"] = "TSLA"
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider({**VALID, "action_plan": broken}), cache_dir=tmp_path)


def test_run_ai_analysis_fails_closed_when_action_plan_exceeds_local_limits(tmp_path, mixed_result):
    """Full path: run_ai_analysis must also fail closed when a schema-valid
    response violates a local business-rule cap (e.g. too many top_actions)."""
    broken = _action_plan(top_actions=["改动一", "改动二", "改动三", "改动四"])
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider({**VALID, "action_plan": broken}), cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Action Plan reduction pass (Step 1): fewer/less-repetitive/more-consistent
# output using existing data only -- see core/ai.py::_prompt. These are
# regression guards, not overfit to one specific demo wording: schema-level
# tests confirm nothing here required *more* items than before, the new
# quality validator is tested narrowly (bare phrase vs. a qualified trigger
# using the same word), and prompt tests assert the guardrail concepts are
# present without pinning exact phrasing.
# ---------------------------------------------------------------------------

def test_action_plan_security_actions_may_be_fewer_than_maximum():
    """D: security_actions must not require more than one entry -- "less but
    meaningful" is a valid plan, not just the historical 3-5 default."""
    parsed = CommitteeResult.model_validate(VALID)
    assert len(parsed.action_plan.security_actions) == 1


def test_action_plan_timeline_accepts_single_observational_entry():
    """G: a timeline horizon with nothing genuinely new to do may contain one
    concise observational entry instead of a fabricated action."""
    broken = _action_plan(timeline_3_months=["暂无新增操作，等待触发条件"])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    assert parsed.action_plan.timeline_3_months == ["暂无新增操作，等待触发条件"]


def test_action_plan_reassessment_trigger_accepts_state_based_condition():
    """H: reassessment_triggers may be a state-based condition (concentration/
    coverage/market-regime) rather than only a calendar-type event."""
    broken = _action_plan(reassessment_triggers=[
        {"event_or_condition": "若集中度持续高于建议区间且未见改善", "affected_holdings": ["NVDA"],
         "reassess": "重新评估集中度管理是否需要加强"},
    ])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    assert parsed.action_plan.reassessment_triggers[0].event_or_condition == "若集中度持续高于建议区间且未见改善"


@pytest.mark.parametrize("phrase", ["逢高", "明显反弹", "适度减仓", "分阶段优化"])
def test_action_plan_rejects_bare_vague_trigger_phrase(mixed_result, phrase):
    """F: a security_action trigger that is nothing but one of the specific
    unqualified vague phrases called out in the reduction pass is rejected --
    it gives the reader no observable condition to act on."""
    from core.ai import _validate_action_plan_facts, _validate_action_plan_quality
    from core.analytics import canonical_fact_packet

    broken = _action_plan()
    broken["security_actions"][0]["trigger"] = phrase
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    _validate_action_plan_facts(parsed.action_plan, canonical_fact_packet(mixed_result))
    with pytest.raises(ValueError, match="vague phrase"):
        _validate_action_plan_quality(parsed.action_plan)


def test_action_plan_accepts_qualified_trigger_containing_vague_word():
    """Companion to the above: the guardrail is narrow -- the same word used
    as part of a concrete, anchored trigger must still pass."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["trigger"] = "若反弹至近期压力区，逢高分阶段减仓"
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    _validate_action_plan_quality(parsed.action_plan)  # must not raise


@pytest.mark.parametrize("pct", [0, -5, 100, 150])
def test_position_reduction_pct_out_of_range_rejected_by_local_validation(pct):
    """Codex P0 fix: since position_reduction_pct carries no Pydantic gt/lt
    (to avoid exclusiveMinimum/exclusiveMaximum in the Gemini-facing
    schema), the strict (0, 100) range must be enforced locally instead.
    action is forced to "REDUCE" here so this specifically exercises the
    range check, not the (also-rejecting, but differently-reasoned)
    Step 2A.1 REDUCE-contract check."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["position_reduction_pct"] = pct
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="out of range"):
        _validate_action_plan_quality(parsed.action_plan)


def test_position_reduction_pct_within_range_passes_local_validation():
    """Updated for Step 2A.1: position_reduction_pct is only ever meaningful
    (and only ever permitted) on an action=="REDUCE" entry."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["position_reduction_pct"] = 10.0
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    _validate_action_plan_quality(parsed.action_plan)  # must not raise


def test_payload_without_position_reduction_pct_field_still_validates():
    """Backward compatibility: an old cached response (or any response that
    simply omits the optional field) for a non-reduction action must
    continue to validate cleanly -- position_reduction_pct defaults to
    None."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "HOLD"
    del broken["security_actions"][0]["position_reduction_pct"]
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    assert parsed.action_plan.security_actions[0].position_reduction_pct is None
    _validate_action_plan_quality(parsed.action_plan)  # must not raise


# ---------------------------------------------------------------------------
# Step 2A.1: REDUCE Action Contract Fix. Production observation: Gemini
# produced action=REDUCE with a target-WEIGHT range (e.g. "目标权重
# 15%-20%") but left position_reduction_pct null -- the local Trade Impact
# Preview correctly failed closed (no preview), but that means the AI
# response itself violated the contract and should never have been treated
# as a valid final answer. These tests lock in: REDUCE always requires a
# usable pct (never derived from target's free text), and a non-REDUCE
# action must never carry one either.
# ---------------------------------------------------------------------------

def test_reduce_with_valid_position_reduction_pct_passes():
    """A."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["position_reduction_pct"] = 20.0
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    _validate_action_plan_quality(parsed.action_plan)  # must not raise


def test_reduce_with_missing_position_reduction_pct_fails_closed():
    """B: this is the exact production bug -- action=REDUCE with pct left
    null must fail closed, not silently render no preview."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["position_reduction_pct"] = None
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="missing position_reduction_pct"):
        _validate_action_plan_quality(parsed.action_plan)


@pytest.mark.parametrize("bad_pct", [float("nan"), float("inf"), float("-inf")])
def test_reduce_with_non_finite_position_reduction_pct_fails_closed(bad_pct):
    """E."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["position_reduction_pct"] = bad_pct
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="out of range"):
        _validate_action_plan_quality(parsed.action_plan)


def test_non_reduce_action_with_null_position_reduction_pct_passes():
    """F: HOLD (or any non-REDUCE action) with pct left null is the normal,
    expected shape and must pass."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "HOLD"
    broken["security_actions"][0]["position_reduction_pct"] = None
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    _validate_action_plan_quality(parsed.action_plan)  # must not raise


def test_non_reduce_action_with_position_reduction_pct_fails_closed():
    """G: a non-REDUCE/REDUCE_ON_REBOUND action carrying a reduction
    percentage is contradictory structured output (the field's only defined
    meaning is "current-position reduction, immediate or trigger-
    contingent") -- fail closed rather than silently ignoring a meaningful
    execution field."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "HOLD"
    broken["security_actions"][0]["position_reduction_pct"] = 20.0
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="only valid for action in"):
        _validate_action_plan_quality(parsed.action_plan)


def test_reduce_with_target_weight_range_text_but_no_pct_fails_closed():
    """H: reproduces the exact production observation -- action=REDUCE,
    target states a target PORTFOLIO WEIGHT range ("目标权重 15%-20%"), and
    position_reduction_pct is left null. Free-text target must never
    substitute for the structured field -- this must fail closed, not
    render silently without a preview."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["target"] = "建议将直接持仓权重分步降至 15%-20% 区间"
    broken["security_actions"][0]["position_reduction_pct"] = None
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="missing position_reduction_pct"):
        _validate_action_plan_quality(parsed.action_plan)


def test_old_cached_payload_without_reduce_action_still_valid():
    """I (no-REDUCE branch): a payload with no REDUCE action anywhere and no
    position_reduction_pct field at all (the pre-Step-2A shape) must remain
    fully valid -- this is exactly VALID itself (action=REDUCE_ON_REBOUND)."""
    from core.ai import _validate_action_plan_quality

    parsed = CommitteeResult.model_validate(VALID)
    _validate_action_plan_quality(parsed.action_plan)  # must not raise


def test_old_cached_payload_with_reduce_action_and_no_pct_field_fails_closed():
    """I (REDUCE branch): an old-shaped payload that happens to have
    action=REDUCE but was produced before this field existed (so the key is
    entirely absent, not just null) must still fail closed -- the contract
    requires an actual usable percentage, not merely "the field is absent
    so we can't check it"."""
    from core.ai import _validate_action_plan_quality

    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    del broken["security_actions"][0]["position_reduction_pct"]
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": broken})
    with pytest.raises(ValueError, match="missing position_reduction_pct"):
        _validate_action_plan_quality(parsed.action_plan)


def test_run_ai_analysis_fails_closed_on_bare_vague_trigger(tmp_path, mixed_result):
    """Full path: a schema-valid, ticker-valid, within-limits response must
    still fail closed when its only trigger is an unqualified vague phrase."""
    broken = _action_plan()
    broken["security_actions"][0]["trigger"] = "明显反弹"
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider({**VALID, "action_plan": broken}), cache_dir=tmp_path)


def test_run_ai_analysis_fails_closed_on_production_reduce_without_pct(tmp_path, mixed_result):
    """Step 2A.1 full-path regression: reproduces the exact production
    observation end-to-end (not just the isolated validator) -- a
    schema-valid, ticker-valid, within-limits REDUCE response with a
    target-weight-range text and no position_reduction_pct must fail
    closed via the existing safe AI-failure path (ProviderUnavailable),
    never silently render an Action Plan with no Trade Impact Preview."""
    broken = _action_plan()
    broken["security_actions"][0]["action"] = "REDUCE"
    broken["security_actions"][0]["target"] = "建议将直接持仓权重分步降至 15%-20% 区间"
    broken["security_actions"][0]["position_reduction_pct"] = None
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider({**VALID, "action_plan": broken}), cache_dir=tmp_path)


def test_run_ai_analysis_succeeds_when_reduce_carries_valid_pct(tmp_path, mixed_result):
    """Manual synthetic contract test (section 11): a REDUCE action with a
    proper position_reduction_pct=20 must validate and succeed end-to-end --
    Gemini only ever supplies the abstract percentage; no raw quantity was
    needed to produce a valid response."""
    fixed = _action_plan()
    fixed["security_actions"][0]["action"] = "REDUCE"
    fixed["security_actions"][0]["position_reduction_pct"] = 20.0
    result, meta = run_ai_analysis(mixed_result, provider=FakeProvider({**VALID, "action_plan": fixed}), cache_dir=tmp_path)
    assert result.action_plan.security_actions[0].position_reduction_pct == 20.0
    assert meta["success"] is True


def test_prompt_includes_coverage_aware_lower_bound_guidance():
    """A: the prompt must instruct the model to treat an incomplete
    look-through true-exposure figure as a lower bound, not a complete fact,
    tied to the actual coverage/uncovered-weight signals already in the
    packet -- not just a generic "be careful" disclaimer."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())  # the source wraps long lines with literal newlines
    assert "lookthrough_coverage" in text
    assert "lookthrough_uncovered_weight" in text
    assert "lower bound" in normalized.lower()
    assert "≥" in text


def test_prompt_does_not_treat_complete_coverage_as_exhaustive():
    """Codex review P1 fix: `lookthrough_coverage == "complete"` only means
    every held ETF has *some* published constituent data (ETF_HOLDINGS is a
    Top-N snapshot, never exhaustive -- see core/reference.py and the ~59.5%
    unnamed residual in VOO's own entry) -- it must never, by itself, permit
    dropping the ">=" lower-bound qualifier for a look-through-derived
    true-exposure figure. The only field-level condition that may drop the
    qualifier is indirect == 0 (a purely direct holding, no ETF look-through
    involved at all)."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert 'complete" and lookthrough_uncovered_weight is ~0 may you state' not in normalized
    assert "indirect contribution is 0" in normalized
    assert "never that 100% of that ETF's holdings are named" in normalized


def test_prompt_includes_decision_confidence_guardrail():
    """B: decision confidence must not exceed data confidence -- a major
    action must not be justified primarily on an incomplete look-through
    number, but direct-data-based risk management is still allowed."""
    from core.ai import _prompt

    text = _prompt({})
    assert "confidence" in text.lower()
    assert "top_direct_holdings" in text
    assert "risk_flags" in text


def test_prompt_distinguishes_ai_target_from_deterministic_fact():
    """C: any numeric target/range is the AI's own recommendation, never a
    deterministically computed threshold -- must be phrased as such."""
    from core.ai import _prompt

    text = _prompt({})
    assert "建议目标" in text or "建议控制区间" in text
    assert "recommendation" in text.lower()


def test_prompt_includes_fund_destination_guardrail():
    """Section 10 (Step 1); strengthened with explicit named-ticker examples
    in Step 2A.2: never claim an unverified destination reduces
    concentration; prefer generic wording over naming an unverified ETF
    such as VOO/SGOV."""
    from core.ai import _prompt

    text = _prompt({})
    assert "转向经穿透验证后确认能够降低集中度的资产" in text
    assert "VOO" in text and "SGOV" in text  # named as forbidden examples, not recommendations


def test_prompt_forbids_complete_lookthrough_coverage_claims():
    """Section 7 (Step 2A.2, L): "complete" coverage must never be described
    as exhaustive/100% look-through -- production regression: "完全穿透覆盖
    （覆盖率约38.39%）" is self-contradictory (a 38% figure cannot be
    "complete"). Prompt must forbid the specific phrases and prefer
    user-friendly, non-jargon wording naming the actual identified
    coverage."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    for forbidden in ("完全穿透覆盖", "100%穿透", "完整真实暴露"):
        assert forbidden in text  # named explicitly as forbidden, i.e. quoted in the guardrail itself
    assert "fully exhaustive" in normalized.lower() or "exhaustive look-through" in normalized.lower()
    assert "已获得穿透数据" in text  # the preferred, user-friendly alternative phrasing


def test_prompt_still_preserves_lower_bound_identified_exposure_semantics():
    """M: the Step 2A.2 coverage-wording addition must not weaken or remove
    Step 1's "≥" lower-bound framing requirement."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "lower bound" in normalized.lower()
    assert "≥" in text
    assert "lookthrough_coverage" in text


def test_prompt_forbids_tax_lot_sequencing_claims():
    """N: no tax-lot history/identification exists -- the AI must never
    claim which lot/batch to sell (production regression: "优先卖出高成本
    份额"). Only a generic "verify actual cost basis" instruction is
    allowed."""
    from core.ai import _prompt

    text = _prompt({})
    assert "优先卖出高成本份额" in text  # named explicitly as the forbidden example
    assert "tax-lot" in text.lower() or "税务摩擦" in text or "成本基础" in text
    assert "减仓前核对实际成本基础与潜在税务影响" in text  # the one allowed phrasing


def test_prompt_prefers_fewer_security_actions():
    """Section 6: "less but meaningful" -- prefer 1-3 over padding to 5."""
    from core.ai import _prompt

    text = _prompt({})
    assert "1-3" in text
    assert "其余核心仓位暂维持不变" in text


def test_prompt_limits_tax_mention_repetition():
    """Section 7: keep tax awareness, but cap repetition across sections."""
    from core.ai import _prompt

    text = _prompt({})
    assert "at most two places" in text


def test_prompt_allows_observational_timeline_entries():
    """Section 8: timeline buckets may hold a concise observational state
    instead of a fabricated action when nothing new is genuinely due."""
    from core.ai import _prompt

    text = _prompt({})
    assert "观察" in text
    assert "暂无新增操作" in text


def test_prompt_allows_state_based_reassessment_triggers():
    """Section 9: reassessment may be state-based, not only calendar events."""
    from core.ai import _prompt

    text = _prompt({})
    assert "state-based" in text.lower()


def test_schema_bytes_match_step_2a_approved_baseline():
    """CommitteeResult's provider-facing JSON Schema must not grow beyond the
    one deliberate, approved Step 2A addition (position_reduction_pct: a
    single flat optional float on SecurityAction, no new $defs) -- any
    further growth should be treated as a new finding, not silently
    rebaselined again."""
    schema = CommitteeResult.model_json_schema()
    schema_bytes = len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    assert schema_bytes == _BASELINE_SCHEMA_BYTES
    assert len(schema.get("$defs", {})) == _BASELINE_SCHEMA_DEFS
    field_schema = schema["$defs"]["SecurityAction"]["properties"]["position_reduction_pct"]
    assert field_schema is not None


def test_position_reduction_pct_schema_has_no_exclusive_bound_keywords():
    """Codex P0 fix: exclusiveMinimum/exclusiveMaximum are outside the
    documented Gemini structured-output subset (this project has hit real
    Gemini 400 INVALID_ARGUMENT failures from unsupported schema keywords
    before). position_reduction_pct must carry no min/max constraint at the
    schema level at all -- the strict (0, 100) range is enforced locally in
    _validate_action_plan_quality instead, after parsing."""
    schema = CommitteeResult.model_json_schema()
    field_schema = json.dumps(schema["$defs"]["SecurityAction"]["properties"]["position_reduction_pct"])
    assert "exclusiveMinimum" not in field_schema
    assert "exclusiveMaximum" not in field_schema
    assert "minimum" not in field_schema
    assert "maximum" not in field_schema


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
    """D: 503 on every one of the 3 primary attempts -- exactly three calls,
    exactly two sleeps (none after the final attempt), fail closed by
    re-raising the ServerError, and the existing final diagnostic logging
    still fires (via run_ai_analysis's stage=generate_content handler).
    GEMINI_FALLBACK_MODEL is pinned equal to GEMINI_MODEL here specifically
    to isolate pure primary-retry-exhaustion behavior from the fallback
    feature (see the same-model guard tests below) -- this is the exact
    pre-fallback-patch scenario/assertions, unchanged."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
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
    with no leaked secrets. GEMINI_FALLBACK_MODEL is pinned equal to
    GEMINI_MODEL to isolate this from the new fallback feature -- see the
    dedicated fallback tests below for the fallback-active scenarios."""
    caplog.set_level(logging.WARNING, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
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


# ---------------------------------------------------------------------------
# Gemini 3.5 Flash fallback: only after the primary exhausts 3 attempts with
# a confirmed 503, never on any other 5xx and never on a 4xx.
# ---------------------------------------------------------------------------

def test_gemini_provider_falls_back_once_after_three_503s(monkeypatch):
    """4, 15: primary 503 x3 exhausted -- exactly one extra call, against
    the default fallback model gemini-3.5-flash (GEMINI_FALLBACK_MODEL
    unset)."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps(VALID, ensure_ascii=False)],
    )

    provider = GeminiProvider()
    text = provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert text == json.dumps(VALID, ensure_ascii=False)
    calls = holder["client"].models.calls
    assert len(calls) == 4
    assert [c["model"] for c in calls] == ["gemini-3.6-flash", "gemini-3.6-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
    assert sleeps == [2, 5]  # no extra sleep before/after the fallback attempt


def test_run_ai_analysis_succeeds_via_fallback_after_three_503s(monkeypatch, caplog, mixed_result, tmp_path):
    """5, 19, 22: full path -- fallback success returns a normal, validated
    CommitteeResult, a sanitized fallback log fires, and the API key is
    never present in any captured log message."""
    caplog.set_level(logging.WARNING, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps(VALID, ensure_ascii=False)],
    )

    result, meta = run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert result.chairman_decision == VALID["chairman_decision"]
    assert meta["provider"] == "gemini"

    messages = [r.getMessage() for r in caplog.records]
    fallback_messages = [m for m in messages if m.startswith("AI provider fallback:")]
    assert len(fallback_messages) == 1
    assert "primary_model=gemini-3.6-flash" in fallback_messages[0]
    assert "fallback_model=gemini-3.5-flash" in fallback_messages[0]
    assert "primary_status=503" in fallback_messages[0]
    assert not any(m.startswith("AI provider failure:") for m in messages)  # no failure -- fallback succeeded
    assert "test-key-not-real" not in "\n".join(messages)


def test_gemini_provider_fails_closed_when_fallback_also_returns_503(monkeypatch, caplog):
    """6: primary 503 x3, fallback also 503 -- fail closed (the fallback's
    own ServerError propagates), and the fallback-specific failure log
    identifies gemini-3.5-flash (not the primary model) as the one that
    actually failed last."""
    from core.ai import GeminiProvider
    from google.genai import errors

    caplog.set_level(logging.WARNING, logger="canam.ai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 4
    messages = [r.getMessage() for r in caplog.records]
    fallback_failure = [m for m in messages if "stage=generate_content_fallback" in m]
    assert len(fallback_failure) == 1
    assert "model=gemini-3.5-flash" in fallback_failure[0]


def test_gemini_provider_fails_closed_when_fallback_returns_400(monkeypatch):
    """7: primary 503 x3, fallback returns a permanent 400 -- fail closed
    immediately (the ClientError propagates as-is, no further retry)."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), _client_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ClientError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 4


def test_gemini_provider_non_503_server_error_does_not_trigger_fallback(monkeypatch):
    """13: a non-503 5xx (e.g. 500 INTERNAL) still retries 3 total primary
    attempts exactly as before, but must NOT trigger a model fallback --
    only a confirmed 503 does."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    internal_error = _server_error(status="INTERNAL", code=500, message="internal error")
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[internal_error, internal_error, internal_error],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 3  # no 4th (fallback) call
    assert sleeps == [2, 5]


def test_gemini_provider_does_not_duplicate_call_when_fallback_equals_primary(monkeypatch):
    """14: if GEMINI_FALLBACK_MODEL == GEMINI_MODEL, do not make a duplicate
    fallback request against the same model -- fail closed right after
    primary exhaustion, exactly 3 calls total."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 3


def test_gemini_fallback_model_env_var_overrides_default(monkeypatch):
    """16: a configured GEMINI_FALLBACK_MODEL is honored instead of the
    gemini-3.5-flash default."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash")
    _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps(VALID, ensure_ascii=False)],
    )

    provider = GeminiProvider()
    assert provider.fallback_model == "gemini-3.7-flash"
    provider.generate("prompt text", CommitteeResult.model_json_schema())
    assert holder["client"].models.calls[3]["model"] == "gemini-3.7-flash"


def test_gemini_fallback_uses_gemini_3_compatible_config_and_same_prompt(monkeypatch):
    """17, 18, 23: the fallback call keeps the exact same structured-output
    config (response_mime_type/response_json_schema, thinking_level
    "minimal", no temperature/thinking_budget) and the exact same prompt
    string as the primary call -- no separate fallback prompt/fact packet."""
    from core.ai import GeminiProvider
    from google.genai import types

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps(VALID, ensure_ascii=False)],
    )

    provider = GeminiProvider()
    schema = CommitteeResult.model_json_schema()
    provider.generate("the exact same prompt text", schema)

    primary_call, fallback_call = holder["client"].models.calls[0], holder["client"].models.calls[3]
    assert fallback_call["contents"] == primary_call["contents"] == "the exact same prompt text"
    config = fallback_call["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert config.response_schema is None
    assert config.temperature is None
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_gemini_fallback_malformed_json_still_fails_closed(monkeypatch, caplog, mixed_result, tmp_path):
    """20: fallback succeeds at the transport level but returns invalid JSON
    -- must still fail closed at response_parse, exactly like a primary-only
    malformed response."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps({"made_up": "facts"})],
    )

    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)
    assert any("stage=response_parse" in r.getMessage() for r in caplog.records)


def test_gemini_fallback_unknown_ticker_action_plan_still_fails_closed(monkeypatch, mixed_result, tmp_path):
    """21: even a fallback response that is well-formed JSON must still go
    through the same Action Plan ticker validation -- an unverified ticker
    fails closed exactly as it would from the primary model."""
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    broken = json.loads(json.dumps(VALID))
    broken["action_plan"]["security_actions"][0]["ticker"] = "TSLA"
    _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps(broken, ensure_ascii=False)],
    )

    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)


@pytest.mark.parametrize("status,code", [("NOT_FOUND", 404), ("RESOURCE_EXHAUSTED", 429)])
def test_gemini_provider_does_not_retry_or_fallback_on_other_client_errors(monkeypatch, status, code):
    """8-12: 404 (invalid/unavailable model) and 429 (quota/rate limit) are
    ClientError like 400/401/403 -- exactly one call, no retry, no
    fallback."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error(status=status, code=code, message="denied")])

    provider = GeminiProvider()
    with pytest.raises(errors.ClientError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 1
    assert sleeps == []


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


# ---------------------------------------------------------------------------
# Production Gemini diagnostic logging: sanitized request-shape/size metadata
# on failure, to let a future production 400 be compared directly against a
# known-good local reproduction. Diagnostic only -- no behavior change.
# ---------------------------------------------------------------------------

def test_400_failure_log_contains_model_length_and_whitespace_flag(monkeypatch, caplog, mixed_result, tmp_path):
    """1, 2: a real 400 (ClientError) failure log includes the exact model
    string's length and whether it carries leading/trailing whitespace."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error()])

    from google.genai import errors
    with pytest.raises(errors.ClientError):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    message = caplog.records[0].getMessage()
    assert "stage=generate_content" in message
    assert f"model_length={len('gemini-3.6-flash')}" in message
    assert "model_has_whitespace=False" in message


def test_400_failure_log_contains_prompt_schema_packet_byte_lengths(monkeypatch, caplog, mixed_result, tmp_path):
    """3, 4, 5: the failure log includes prompt/schema/fact-packet UTF-8
    byte lengths -- never the content itself."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error()])

    from google.genai import errors
    with pytest.raises(errors.ClientError):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    message = caplog.records[0].getMessage()
    for field in ("prompt_bytes=", "schema_bytes=", "packet_bytes="):
        assert field in message
        value = message.split(field, 1)[1].split(" ", 1)[0]
        assert int(value) > 0


def test_400_failure_log_contains_api_key_presence_length_whitespace(monkeypatch, caplog, mixed_result, tmp_path):
    """6: the failure log reports GEMINI_API_KEY presence/length/whitespace
    -- metadata only, never the key value."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error()])

    from google.genai import errors
    with pytest.raises(errors.ClientError):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    message = caplog.records[0].getMessage()
    assert "gemini_key_present=True" in message
    assert f"gemini_key_length={len('test-key-not-real')}" in message
    assert "gemini_key_has_whitespace=False" in message


def test_400_failure_log_never_contains_key_prompt_packet_or_holdings(monkeypatch, caplog, mixed_result, tmp_path):
    """7, 8, 9, 10: no API key value, prompt text, fact-packet content, or
    holding/ticker names ever appear in the failure log -- only the
    sanitized size/presence metadata added above."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    real_key = "AIzaSyREALFAKEKEYVALUE1234567890"
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", real_key)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    _no_sleep_calls(monkeypatch)
    _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error()])

    from google.genai import errors
    with pytest.raises(errors.ClientError):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    message = caplog.records[0].getMessage()
    assert real_key not in message
    assert "NVDA" not in message and "VOO" not in message and "SGOV" not in message
    assert "top_direct_holdings" not in message and "top_true_exposures" not in message
    assert "portfolio_score" not in message
    # The instructional prompt template text must not leak either.
    assert "Seven-Member Investment Committee" not in message


def test_400_still_does_not_retry_or_fallback_with_new_logging(monkeypatch, mixed_result, tmp_path):
    """11, 12, 13: the new logging is purely additive -- a 400 still makes
    exactly one call, no retry sleep, and no fallback attempt."""
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(monkeypatch, outcomes=[_client_error()])

    from google.genai import errors
    with pytest.raises(errors.ClientError):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    assert len(holder["client"].models.calls) == 1
    assert sleeps == []


def test_success_path_emits_no_diagnostic_failure_log_with_new_fields(caplog, mixed_result, tmp_path):
    """14: a successful run still emits zero failure-log records -- the new
    fields only ever appear on a failure path, never as a per-request
    success log."""
    caplog.set_level(logging.ERROR, logger="canam.ai")
    run_ai_analysis(mixed_result, provider=FakeProvider(), cache_dir=tmp_path)
    assert caplog.records == []


def test_fallback_failure_log_includes_new_fields_and_marks_fallback_active(monkeypatch, mixed_result, tmp_path):
    """Section 4/7: when the fallback model itself fails, the
    stage=generate_content_fallback log (emitted from inside GeminiProvider)
    also carries prompt/schema byte lengths, fallback_active=True, and an
    attempt count -- without changing the existing fallback-failure behavior
    (still exactly one extra call, still fails closed)."""
    import logging as _logging
    logger = _logging.getLogger("canam.ai")
    records = []

    class _Handler(_logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Handler()
    logger.addHandler(handler)
    logger.setLevel(_logging.WARNING)
    try:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
        monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
        _no_sleep_calls(monkeypatch)
        _install_flaky_gemini_client(
            monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), _server_error()],
        )
        from core.ai import GeminiProvider
        from google.genai import errors

        provider = GeminiProvider()
        with pytest.raises(errors.ServerError):
            provider.generate("prompt text", CommitteeResult.model_json_schema())
    finally:
        logger.removeHandler(handler)

    fallback_failure = [r for r in records if "stage=generate_content_fallback" in r.getMessage()]
    assert len(fallback_failure) == 1
    message = fallback_failure[0].getMessage()
    assert "fallback_active=True" in message
    assert "attempt=4" in message  # 3 exhausted primary attempts + 1 fallback attempt
    assert "prompt_bytes=" in message and "schema_bytes=" in message
    assert "prompt text" not in message  # the actual prompt content must not leak
