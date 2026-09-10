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
# Rebaselined a third time in Step 2A.3's P1-1 follow-up: ActionPlan.
# security_actions dropped its minItems=1 constraint (a genuinely no-action
# portfolio may now return an empty list -- see _validate_action_plan_quality
# and the Step 2A.3 report) -- a 15-byte *decrease*, no new $defs.
# Rebaselined a fourth time in Step 2A.10.1: CommitteeMember.stance's enum
# dropped "维持配置" ("maintain configuration") -- a production regression
# showed the model using it as a specialist verdict LABEL to endorse the
# current overall allocation, which this product's Layer 1 data never
# proves (see the no-action-vs-maintain-allocation rule in core.ai._prompt
# and _MAINTAIN_ALLOCATION_PHRASES). Removed at the schema level so the
# model cannot structurally select it -- "暂缓行动" (defer action) already
# covers the correct "no current trigger" semantic, so no replacement
# value was added. A 16-byte *decrease*, no new $defs.
_BASELINE_SCHEMA_BYTES = 4072
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
    in Step 2A.2. Step 2A.8 retires the prior "use general wording instead"
    escape hatch entirely -- with no correlation model and no post-trade
    destination simulation, NO destination (named ticker or generic type) is
    ever recommended; proceeds default to hypothetical cash instead."""
    from core.ai import _prompt

    text = _prompt({})
    assert "转向经穿透验证后确认能够降低集中度的资产" in text  # named as a forbidden example, not a recommendation
    assert "VOO" in text and "SGOV" in text  # named as forbidden examples, not recommendations
    assert "卖出所得暂存为现金，不预设再投资标的" in text
    assert "低相关性资产" in text  # named as forbidden -- no correlation model exists


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
    """Section 6: "less but meaningful" -- prefer 1-3 over padding to 5.
    Step 2A.8: the closing filler line for untouched holdings must no
    longer claim they are being "maintained" (an unsupported target-
    allocation claim) -- it must say no other holding was identified as
    needing priority action instead."""
    from core.ai import _prompt

    text = _prompt({})
    assert "1-3" in text
    assert "当前已验证数据未识别出需要优先处理的其他持仓" in text
    assert "其余核心仓位暂维持不变" in text  # named explicitly as the forbidden example, not a recommendation


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
    """CommitteeResult's provider-facing JSON Schema must match the current
    approved baseline (see _BASELINE_SCHEMA_BYTES for the full rebaseline
    history, most recently Step 2A.3's deliberate, approved removal of
    security_actions' minItems=1) -- any further drift should be treated as
    a new finding, not silently rebaselined again."""
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


def test_gemini_provider_fails_closed_when_fallback_exhausts_both_attempts_with_503(monkeypatch, caplog):
    """6; Gemini reliability patch TESTS REQUIRED #3: primary 503 x3,
    fallback attempt 1 also 503 -- the reliability patch retries once more
    (waiting ~4s) instead of failing immediately, and fallback attempt 2 also
    503 -- NOW fails closed (the fallback's own ServerError propagates)
    after exactly 5 total calls (3 primary + 2 fallback), and the
    fallback-specific final-failure log identifies gemini-3.5-flash (not the
    primary model) as the one that actually failed last."""
    from core.ai import GeminiProvider
    from google.genai import errors

    caplog.set_level(logging.WARNING, logger="canam.ai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch,
        outcomes=[_server_error(), _server_error(), _server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 5  # 3 primary + 2 fallback attempts
    assert sleeps == [2, 5, 4]  # primary backoff unchanged, plus one ~4s fallback-retry wait
    messages = [r.getMessage() for r in caplog.records]
    fallback_retry = [m for m in messages if "transient failure" in m and "stage=generate_content_fallback" in m]
    fallback_final_failure = [
        m for m in messages if m.startswith("AI provider failure:") and "stage=generate_content_fallback" in m
    ]
    assert len(fallback_retry) == 1
    assert "attempt=1/2" in fallback_retry[0]
    assert len(fallback_final_failure) == 1
    assert "model=gemini-3.5-flash" in fallback_final_failure[0]
    assert "attempt=5" in fallback_final_failure[0]  # 3 primary + 2 fallback attempts


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


# ---------------------------------------------------------------------------
# Gemini reliability patch: the fallback model itself now gets a bounded
# second attempt (at most 2 total fallback calls) instead of exactly one.
# Fallback attempt 2 fires only if attempt 1 fails with a confirmed 503,
# after a fixed ~4s wait -- any non-503 fallback failure still fails closed
# immediately, exactly as before. Primary retry count/timing/conditions and
# fallback *activation* conditions (confirmed 503 after 3 exhausted primary
# attempts, distinct fallback model) are entirely unchanged -- see the tests
# above this section.
# ---------------------------------------------------------------------------

def test_fallback_attempt_one_succeeds_existing_behavior_preserved(monkeypatch):
    """TESTS REQUIRED #1: primary 503 exhaustion -> fallback attempt 1
    succeeds -- identical to pre-patch behavior, exactly 4 total calls, no
    fallback-retry wait."""
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
    assert len(holder["client"].models.calls) == 4  # 3 primary + fallback attempt 1
    assert sleeps == [2, 5]  # no fallback-retry wait needed


def test_fallback_attempt_one_503_then_attempt_two_succeeds(monkeypatch):
    """TESTS REQUIRED #2: primary 503 exhaustion -> fallback attempt 1
    returns 503 -> waits ~4s -> fallback attempt 2 succeeds. Exactly 5 total
    calls, same model/prompt/schema/config on both fallback attempts."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch,
        outcomes=[
            _server_error(), _server_error(), _server_error(),
            _server_error(), json.dumps(VALID, ensure_ascii=False),
        ],
    )

    provider = GeminiProvider()
    text = provider.generate("the exact same prompt text", CommitteeResult.model_json_schema())

    assert text == json.dumps(VALID, ensure_ascii=False)
    calls = holder["client"].models.calls
    assert len(calls) == 5  # 3 primary + 2 fallback attempts
    assert [c["model"] for c in calls] == [
        "gemini-3.6-flash", "gemini-3.6-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash",
    ]
    assert sleeps == [2, 5, 4]  # primary backoff unchanged, plus exactly one ~4s fallback-retry wait
    # Same prompt/schema/config on both fallback attempts (18, 23 carried forward).
    fallback_call_1, fallback_call_2 = calls[3], calls[4]
    assert fallback_call_1["contents"] == fallback_call_2["contents"] == "the exact same prompt text"
    assert fallback_call_1["config"].response_json_schema == fallback_call_2["config"].response_json_schema


def test_fallback_both_attempts_503_final_clean_failure(monkeypatch):
    """TESTS REQUIRED #3: primary 503 exhaustion -> fallback attempt 1
    returns 503 -> fallback attempt 2 also returns 503 -- final clean
    failure (the fallback's own ServerError propagates), exactly 5 total
    calls, no infinite loop. See test_gemini_provider_fails_closed_when_
    fallback_exhausts_both_attempts_with_503 above for the full logging
    assertions on this same scenario."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch,
        outcomes=[_server_error(), _server_error(), _server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 5  # exactly 5, no infinite loop
    assert sleeps == [2, 5, 4]


def test_fallback_non_503_server_error_no_second_attempt(monkeypatch):
    """TESTS REQUIRED #4: fallback attempt 1 fails with a non-503 5xx (e.g.
    500 INTERNAL) -- must fail closed immediately, no second fallback
    attempt, no fallback-retry wait."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    fallback_internal_error = _server_error(status="INTERNAL", code=500, message="internal error")
    holder = _install_flaky_gemini_client(
        monkeypatch,
        outcomes=[_server_error(), _server_error(), _server_error(), fallback_internal_error],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError) as excinfo:
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert excinfo.value.code == 500
    assert len(holder["client"].models.calls) == 4  # 3 primary + exactly 1 fallback attempt
    assert sleeps == [2, 5]  # no fallback-retry wait for a non-503


def test_fallback_client_error_4xx_no_second_attempt(monkeypatch):
    """TESTS REQUIRED #5: fallback attempt 1 fails with a ClientError/4xx --
    must fail closed immediately, no second fallback attempt. See also
    test_fallback_failure_log_includes_new_fields_and_marks_fallback_active
    for the full logging assertions on this same scenario."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), _client_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ClientError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 4  # 3 primary + exactly 1 fallback attempt
    assert sleeps == [2, 5]


def test_fallback_malformed_response_no_second_attempt(monkeypatch, mixed_result, tmp_path):
    """TESTS REQUIRED #6: fallback attempt 1 succeeds at the transport level
    but returns malformed/invalid JSON -- this is not a ServerError at all
    (the SDK call itself succeeded), so the reliability patch's 503-only
    retry never engages -- exactly one fallback attempt, still fails closed
    at response_parse via run_ai_analysis, exactly like before this patch."""
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), json.dumps({"made_up": "facts"})],
    )

    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, cache_dir=tmp_path)

    assert len(holder["client"].models.calls) == 4  # 3 primary + exactly 1 fallback attempt
    assert sleeps == [2, 5]


def test_maximum_provider_call_count_is_five(monkeypatch):
    """TESTS REQUIRED #7: the absolute worst case (3 primary 503s + 2
    fallback 503s) makes exactly 5 provider calls total -- never more, no
    infinite loop, regardless of how many more 503s a real flaky client
    might otherwise produce."""
    from core.ai import GeminiProvider
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
    _no_sleep_calls(monkeypatch)
    # Exactly 5 outcomes provided -- if the implementation ever made a 6th
    # call, _FlakyGeminiModels.generate_content would raise IndexError
    # ("pop from empty list") instead of the expected ServerError, failing
    # this test with the wrong exception type.
    holder = _install_flaky_gemini_client(
        monkeypatch,
        outcomes=[_server_error(), _server_error(), _server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(errors.ServerError):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 5


def test_primary_retry_schedule_remains_exactly_unchanged(monkeypatch):
    """TESTS REQUIRED #8: the Gemini reliability patch touches only the
    fallback path -- 3 total primary attempts, waits of exactly 2s then 5s,
    unaffected by whether a fallback is later engaged."""
    from core.ai import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")  # isolate primary-only behavior
    sleeps = _no_sleep_calls(monkeypatch)
    holder = _install_flaky_gemini_client(
        monkeypatch, outcomes=[_server_error(), _server_error(), _server_error()],
    )

    provider = GeminiProvider()
    with pytest.raises(Exception):
        provider.generate("prompt text", CommitteeResult.model_json_schema())

    assert len(holder["client"].models.calls) == 3
    assert sleeps == [2, 5]


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
    """Section 4/7; Gemini reliability patch TESTS REQUIRED #5 (fallback
    ClientError/4xx -> no second fallback attempt): when the fallback model's
    one attempt fails with a non-503 (a 4xx ClientError never qualifies for
    the reliability patch's bounded 503-only retry), the
    stage=generate_content_fallback log (emitted from inside GeminiProvider)
    still carries prompt/schema byte lengths, fallback_active=True, and an
    attempt count -- exactly one extra call, still fails closed, no retry."""
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
        sleeps = _no_sleep_calls(monkeypatch)
        holder = _install_flaky_gemini_client(
            monkeypatch, outcomes=[_server_error(), _server_error(), _server_error(), _client_error()],
        )
        from core.ai import GeminiProvider
        from google.genai import errors

        provider = GeminiProvider()
        with pytest.raises(errors.ClientError):
            provider.generate("prompt text", CommitteeResult.model_json_schema())
    finally:
        logger.removeHandler(handler)

    assert len(holder["client"].models.calls) == 4  # 3 primary + exactly 1 fallback attempt, no retry
    assert sleeps == [2, 5]  # no fallback-retry wait -- non-503 fails closed immediately
    fallback_failure = [r for r in records if "stage=generate_content_fallback" in r.getMessage()]
    assert len(fallback_failure) == 1
    message = fallback_failure[0].getMessage()
    assert "fallback_active=True" in message
    assert "attempt=4" in message  # 3 exhausted primary attempts + 1 fallback attempt
    assert "prompt_bytes=" in message and "schema_bytes=" in message
    assert "prompt text" not in message  # the actual prompt content must not leak


# ============================================================================
# Step 2A.3 -- Report Integrity Guardrails
#
# Four production regressions: (P0-1) absence of a triggered risk_flag was
# reworded as "risk controlled"; (P0-2) direct_effective_n (direct-holdings
# concentration only) was reworded as overall diversification; (P1-1) the
# Action Plan padded HOLD/WAIT into high-priority slots and repeated
# observational filler across every timeline horizon; (P1-2) a current fact
# (cash weight) was echoed back as a recommended target with no independent
# justification. See core.ai._validate_risk_confidence_language for the one
# new fail-closed validator this step adds; everything else is prompt-only.
# ============================================================================

_MATERIAL_UNCERTAINTY_PACKET = {"data_quality": {"lookthrough_coverage": "insufficient", "lookthrough_uncovered_weight": 0.3494}}
_COMPLETE_COVERAGE_PACKET = {"data_quality": {"lookthrough_coverage": "complete", "lookthrough_uncovered_weight": 0.0}}


def _committee(**overrides):
    payload = json.loads(json.dumps(VALID))
    payload.update(overrides)
    return payload


# --- Unit tests: the narrow negation-aware phrase detector -----------------

def test_unnegated_overconfident_phrase_flags_bare_claim():
    """P0-1 production regression, verbatim: an unqualified "风险受控" claim
    must be detected."""
    from core.ai import _unnegated_overconfident_phrase

    assert _unnegated_overconfident_phrase("鉴于目前组合风险受控且无紧急风险警报，维持现状。") == "风险受控"


def test_unnegated_overconfident_phrase_ignores_negated_claim():
    """Part 6 C: a sentence that DENIES the overconfident claim must never be
    rejected merely for containing the same words -- "不能确认风险受控" is the
    correct, required framing, not a violation."""
    from core.ai import _unnegated_overconfident_phrase

    assert _unnegated_overconfident_phrase("不能确认风险受控") is None
    assert _unnegated_overconfident_phrase("当前已验证数据未触发新的确定性风险警报，但穿透覆盖有限，不能据此确认整体风险较低。") is None


def test_unnegated_overconfident_phrase_detects_p02_direct_effective_n_variant():
    """P0-2 production regression, verbatim: Direct Effective N was used to
    claim the portfolio is "处于受控范围"."""
    from core.ai import _unnegated_overconfident_phrase

    assert _unnegated_overconfident_phrase("组合直接有效分散户数约为5.76，处于受控范围") == "处于受控范围"


def test_unnegated_overconfident_phrase_none_for_clean_text():
    from core.ai import _unnegated_overconfident_phrase

    assert _unnegated_overconfident_phrase("直接持仓口径 Effective N 约为5.76，仅反映直接持仓分布。") is None


# --- Unit tests: materiality threshold --------------------------------------

def test_lookthrough_uncertainty_is_material_for_partial_and_insufficient_coverage():
    from core.ai import _lookthrough_uncertainty_is_material

    assert _lookthrough_uncertainty_is_material({"data_quality": {"lookthrough_coverage": "partial", "lookthrough_uncovered_weight": 0.01}})
    assert _lookthrough_uncertainty_is_material({"data_quality": {"lookthrough_coverage": "insufficient", "lookthrough_uncovered_weight": 0.5}})


def test_lookthrough_uncertainty_is_material_false_for_complete_coverage_and_low_uncovered_weight():
    from core.ai import _lookthrough_uncertainty_is_material

    assert not _lookthrough_uncertainty_is_material({"data_quality": {"lookthrough_coverage": "complete", "lookthrough_uncovered_weight": 0.0}})
    assert not _lookthrough_uncertainty_is_material({"data_quality": {"lookthrough_coverage": "complete", "lookthrough_uncovered_weight": 0.02}})


def test_lookthrough_uncertainty_is_material_true_above_uncovered_weight_threshold_even_if_labelled_complete():
    """Same >0.05 threshold the prompt already asks the model to key its own
    coverage-aware wording off of (see _prompt) -- the validator must agree."""
    from core.ai import _lookthrough_uncertainty_is_material

    assert _lookthrough_uncertainty_is_material({"data_quality": {"lookthrough_coverage": "complete", "lookthrough_uncovered_weight": 0.06}})


# --- Unit tests: the fail-closed validator itself ---------------------------

def test_validate_risk_confidence_language_fails_closed_on_majority_view():
    from core.ai import _validate_risk_confidence_language

    committee = CommitteeResult.model_validate(_committee(majority_view="鉴于目前组合风险受控且无紧急风险警报，维持现状。"))
    with pytest.raises(ValueError, match="风险受控"):
        _validate_risk_confidence_language(committee, _MATERIAL_UNCERTAINTY_PACKET)


def test_validate_risk_confidence_language_scans_non_risk_specialists_too():
    """P0-1: the guardrail is not risk-role-specific -- any specialist
    (macro/portfolio/valuation_data/tax/action_rebalancing) making the same
    overconfident claim must also fail closed."""
    from core.ai import _validate_risk_confidence_language

    payload = _committee()
    for member in payload["members"]:
        if member["role"] == "portfolio":
            member["conclusion"] = "组合风险受控，无需调整。"
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="members\\[portfolio\\]"):
        _validate_risk_confidence_language(committee, _MATERIAL_UNCERTAINTY_PACKET)


def test_validate_risk_confidence_language_allows_negated_phrasing():
    """The confidence-qualified, allowed framing must pass unchanged."""
    from core.ai import _validate_risk_confidence_language

    committee = CommitteeResult.model_validate(
        _committee(majority_view="当前已验证数据未触发新的确定性风险警报，但穿透覆盖有限，不能据此确认整体风险较低。")
    )
    _validate_risk_confidence_language(committee, _MATERIAL_UNCERTAINTY_PACKET)  # must not raise


def test_validate_risk_confidence_language_skips_when_coverage_is_complete():
    """The validator must never second-guess a well-covered portfolio's
    legitimate reporting -- only fires when coverage is materially
    incomplete."""
    from core.ai import _validate_risk_confidence_language

    committee = CommitteeResult.model_validate(_committee(majority_view="当前组合风险受控。"))
    _validate_risk_confidence_language(committee, _COMPLETE_COVERAGE_PACKET)  # must not raise


# --- Full-path integration tests via run_ai_analysis ------------------------

def test_run_ai_analysis_fails_closed_on_overconfident_risk_claim_with_incomplete_coverage(tmp_path, low_confidence_result):
    """Scenario 1 (Part 9): a portfolio with a materially incomplete
    look-through ETF and zero deterministic risk_flags must still fail
    closed if the AI phrases that absence of a flag as "risk controlled"."""
    broken = _committee(majority_view="鉴于目前组合风险受控且无紧急风险警报，维持现状。")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(low_confidence_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_run_ai_analysis_succeeds_with_confidence_qualified_language_under_incomplete_coverage(tmp_path, low_confidence_result):
    """The required allowed framing -- detected risk vs. unverified risk --
    must pass end-to-end against the same low-confidence portfolio."""
    ok = _committee(majority_view="当前已验证数据未触发新的确定性风险警报，但穿透覆盖有限，不能据此确认整体风险较低。")
    parsed, meta = run_ai_analysis(low_confidence_result, provider=FakeProvider(ok), cache_dir=tmp_path)
    assert meta["success"]


def test_run_ai_analysis_does_not_reject_overconfident_phrase_when_coverage_is_complete(tmp_path, mixed_result):
    """Guardrail scope check: the same phrase that fails closed under
    materially incomplete coverage must not be rejected against a portfolio
    whose look-through coverage is already complete (mixed_result's VOO has
    published constituent data) -- the validator is coverage-conditional, not
    a blanket ban on the phrase."""
    same_phrase = _committee(majority_view="当前组合风险受控。")
    parsed, meta = run_ai_analysis(mixed_result, provider=FakeProvider(same_phrase), cache_dir=tmp_path)
    assert meta["success"]


# --- Prompt-regression tests: P0-1 risk-confidence wording ------------------

def test_prompt_distinguishes_detected_risk_from_unverified_risk():
    """B: the prompt must explicitly instruct the model that absence of a
    triggered deterministic flag is not itself proof of safety."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "risk_flags" in text
    assert "is NOT itself evidence" in normalized or "not itself evidence" in normalized.lower()
    assert "当前已验证数据未触发新的确定性风险警报" in text


def test_prompt_forbids_risk_controlled_phrases_under_material_uncertainty():
    """A: the specific production-regression phrases must be named as
    forbidden in the prompt when coverage is materially incomplete."""
    from core.ai import _prompt

    text = _prompt({})
    for forbidden in ("风险受控", "整体风险完全受控", "没有风险", "整体风险较低", "处于受控范围"):
        assert forbidden in text


def test_prompt_risk_confidence_guardrail_applies_to_every_specialist_and_chairman():
    """P0-1 + Part 5: the same wording rule must cover every committee
    member, not only the risk role, and must bind majority_view/main_concern/
    chairman_decision so the chairman cannot upgrade a specialist's stated
    uncertainty into unqualified certainty."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "majority_view" in text and "main_concern" in text and "chairman_decision" in text
    assert "not only the risk role" in normalized


def test_prompt_preserves_deterministic_risk_level_unchanged():
    """Explicit guard against the oversimplified fix this step forbids
    (Part 1): coverage limitations must not force risk_level to Medium+."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "Do NOT change risk_level itself" in normalized or "risk_level itself" in normalized


# --- Prompt-regression tests: P0-2 Direct Effective N scope -----------------

def test_prompt_direct_effective_n_scoped_to_direct_holdings():
    from core.ai import _prompt

    text = _prompt({})
    assert "direct_effective_n" in text
    assert "直接持仓口径" in text


def test_prompt_forbids_direct_effective_n_overall_diversification_claims():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    for forbidden in ("overall diversification", "true diversification", "factor diversification", "economic diversification", "sufficiently diversified"):
        assert forbidden in normalized.lower()


def test_prompt_keeps_direct_and_lookthrough_effective_n_semantically_distinct():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "direct_effective_n" in text and "lookthrough_effective_n" in text
    assert "never conflate" in normalized.lower()


def test_prompt_forbids_factor_effective_n():
    """Part 7 / Part 2: this patch must never introduce or suggest a
    factor-adjusted or correlation-adjusted Effective N."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split()).lower()
    assert "never compute, name, or imply any" in normalized
    assert "factor-adjusted" in normalized or 'factor" ' in normalized


# --- Prompt-regression tests: P1-1 Action Plan reduction --------------------

def test_prompt_forbids_high_priority_hold_padding():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "HOLD or WAIT must not occupy priority" in normalized


def test_prompt_forbids_fabricated_actions_to_fill_slots():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "Never fabricate a REDUCE, ADD, or other action merely to fill a slot" in normalized


def test_prompt_discourages_repetitive_timeline_filler():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "do not repeat the same generic" in normalized
    assert "information density" in normalized.lower()


# --- Prompt-regression tests: P1-2 current fact vs. recommended target ------

def test_prompt_current_fact_is_not_automatically_a_target():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "is a FACT about today's portfolio, never automatically" in normalized
    assert "当前现金约0.38%" in text


def test_prompt_cash_target_requires_independent_justification_and_ai_label():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "建议保持在该比例左右" in text  # named as the forbidden unjustified restatement
    assert '"建议"/"AI建议"' in text or "AI建议" in text


# ============================================================================
# Step 2A.4 -- Report Semantic Consistency
#
# A narrow prompt-only patch correcting four misleading terminology/inference
# patterns the approved Codex read-only audit found: (A) lookthrough_covered_
# weight/uncovered_weight described as constituent-level look-through
# coverage rather than "portfolio weight in equity ETFs with/without some
# reference holdings data"; (B) those two figures treated as complements
# summing to 100%; (C) valuation_coverage (a priced-position COUNT ratio)
# described as fundamental valuation-data coverage; (D) the Fixed Income
# bucket (which includes cash-like Treasury ETFs like SGOV) collapsed into
# plain "债券"; (E) current SGOV/Fixed-Income weight alone treated as proof
# of an optimal target or "stability cornerstone". No Layer 1 formula,
# packet field, or JSON schema changes -- see tests/test_analytics.py for the
# formula-level pins and test_schema_bytes_match_step_2a_3_approved_baseline
# above for the unchanged schema.
# ============================================================================

# --- A: coverage labels / prompt semantics ----------------------------------

def test_prompt_covered_weight_not_described_as_exhaustive_coverage():
    """A1: lookthrough_covered_weight must be framed as "portfolio weight in
    equity ETFs with some reference data", never as identified/exhaustive
    constituent coverage."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "lookthrough_covered_weight" in text
    assert "有参考持仓数据的股票 ETF 占组合比例" in text
    assert "neither is a constituent-identification coverage percentage" in normalized


def test_prompt_uncovered_weight_described_as_no_reference_data():
    """A2: lookthrough_uncovered_weight must be framed explicitly as equity
    ETF portfolio weight with NO reference holdings data."""
    from core.ai import _prompt

    text = _prompt({})
    assert "lookthrough_uncovered_weight" in text
    assert "暂无参考持仓数据的股票 ETF 占组合比例" in text


def test_prompt_forbids_treating_covered_and_uncovered_as_complements():
    """A3: 34.43%-style covered and 8.98%-style uncovered figures are
    disjoint slices of the equity-ETF allocation, never complements of each
    other or of 100% -- the prompt must explicitly forbid computing
    "uncovered = 100% - covered_weight" or summing the two as total
    coverage."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "NEVER complements of each other or of 100%" in normalized
    assert '"uncovered = 100% - covered_weight"' in normalized
    assert "disjoint slices of the equity-ETF allocation" in normalized


def test_prompt_complete_coverage_cannot_imply_exhaustive_constituent_coverage():
    """A4: lookthrough_coverage == "complete" must be framed as "every held
    equity ETF has at least some reference data", never as full constituent
    identification -- extends the existing Step 2A.2 guardrail with the
    exact Step 2A.4 preferred Chinese phrasing."""
    from core.ai import _prompt

    text = _prompt({})
    assert "所有持有的股票 ETF 均有部分参考持仓数据" in text
    assert "底层资产全部覆盖" in text  # named explicitly as forbidden
    assert "已完全识别底层持仓" in text  # named explicitly as forbidden


# --- B: lower-bound semantics (regression guard) ----------------------------

def test_prompt_lower_bound_exposure_semantics_survive_step_2a_4():
    """5: Step 2A.4's new coverage-semantic paragraphs must not weaken or
    remove the pre-existing "≥" lower-bound requirement for ETF-derived
    true-exposure figures (audit Finding D: NVDA ≥15.7%-style wording)."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "≥" in text
    assert "lower bound" in normalized.lower()
    assert "verified LOWER" in text or "verified lower" in normalized.lower()


# --- C: market-value coverage ------------------------------------------------

def test_prompt_valuation_coverage_uses_market_value_calculation_wording():
    """8: the AI must call this figure 持仓市值计算覆盖率 (or clearly
    equivalent wording), never a valuation-data coverage claim."""
    from core.ai import _prompt

    text = _prompt({})
    assert "valuation_coverage" in text
    assert "持仓市值计算覆盖率" in text


def test_prompt_forbids_fundamental_valuation_coverage_wording():
    """9: the prompt must explicitly forbid describing valuation_coverage as
    "估值数据覆盖率"/"估值覆盖率"/fundamental valuation-data coverage, and
    must state plainly that it is a position-count pricing ratio, and that no
    fundamental valuation metric (P/E, P/B, yield, ...) exists in the
    packet."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "估值数据覆盖率" in text  # named explicitly as forbidden
    assert "估值覆盖率" in text  # named explicitly as forbidden
    assert "valuation data coverage" in normalized.lower()
    assert "position-count" in normalized.lower()
    assert "never invent one" in normalized.lower()


# --- D: asset allocation -----------------------------------------------------

def test_prompt_fixed_income_described_as_bonds_and_cash_like():
    """11: the Fixed Income bucket must be presented to ordinary users as
    债券与现金类 (Bonds & Cash-like), reflecting that it includes cash-like
    Treasury ETFs (SGOV/CBIL), not a pure-bond category."""
    from core.ai import _prompt

    text = _prompt({})
    assert "债券与现金类" in text
    assert "SGOV" in text and "CBIL" in text


def test_prompt_forbids_collapsing_fixed_income_into_plain_bonds():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    for forbidden in ("债券", "纯债券", "传统固定收益"):
        assert forbidden in text  # named explicitly as forbidden standalone framing
    assert '"股债"' in text or "股债" in text  # the forbidden stock/bond dichotomy framing


def test_prompt_keeps_cash_and_other_separate_from_fixed_income():
    """12: actual Cash/Other must be named as their own separate figure, not
    folded into the Fixed Income ("Bonds & Cash-like") figure."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "never fold Cash into the Fixed Income figure" in normalized.lower() or "never fold cash into the fixed income figure" in normalized.lower()


def test_prompt_forbids_forcing_allocation_percentages_to_sum_to_100():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "never invent or reallocate a rounding residual" in normalized.lower()


# --- E: current fact != target, SGOV-specific -------------------------------

def test_prompt_sgov_current_weight_alone_cannot_justify_target_or_cornerstone():
    """14: current SGOV/Fixed-Income weight alone must not justify "maintain
    current target", "current allocation optimal", or "stability
    cornerstone" -- the exact production-regression phrases must be named as
    forbidden."""
    from core.ai import _prompt

    text = _prompt({})
    for forbidden in (
        "保持当前SGOV配置", "维持SGOV现有比例", "SGOV作为组合稳定基石",
        "当前SGOV配置最优", "当前固收比例应保持不变",
    ):
        assert forbidden in text
    normalized = " ".join(text.split())
    assert "CURRENT WEIGHT ALONE IS NOT SUFFICIENT EVIDENCE" in normalized


def test_prompt_allows_independently_reasoned_sgov_hold():
    """15: the SGOV/Fixed-Income guardrail must not ban HOLD/CONTROL_
    ADDITIONS/WAIT outright -- those remain legitimate when reasoned from
    packet-visible facts and labeled as the AI's own recommendation."""
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "does NOT ban HOLD, CONTROL_ADDITIONS, or WAIT" in normalized
    assert "legitimate whenever reasoned from packet-visible facts" in normalized


def test_prompt_forbids_manufactured_sgov_hold_card():
    """Part 6: the Action Plan must not manufacture an SGOV/Fixed-Income HOLD
    card merely because the holding exists -- ties the Step 2A.3 no-filler
    rule to the Step 2A.4 SGOV example explicitly."""
    from core.ai import _prompt

    text = _prompt({})
    assert "never manufacture an SGOV or other Fixed-Income HOLD" in text


def test_security_actions_may_still_be_empty_after_step_2a_4():
    """16: Step 2A.3's empty-security_actions cardinality relaxation is
    untouched by this prompt-only patch."""
    plan = _action_plan(security_actions=[])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": plan})
    assert parsed.action_plan.security_actions == []


# --- Structural / schema tests -----------------------------------------------

def test_hold_action_may_use_low_priority_without_rejection():
    """G: nothing in the schema or local validation forces HOLD/WAIT into a
    high-priority slot -- a low-priority HOLD must parse and validate fine."""
    plan = _action_plan()
    plan["security_actions"][0].update({"action": "HOLD", "priority": "低", "position_reduction_pct": None})
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": plan})
    assert parsed.action_plan.security_actions[0].priority == "低"


# --- P1-1 follow-up: security_actions may now be an empty list -------------
#
# Phase 0 originally found ActionPlan.security_actions required min_length=1,
# structurally forcing a fabricated HOLD/WAIT entry even when no holding
# genuinely needed one. This follow-up drops that floor (see the
# security_actions field in core.ai.ActionPlan) -- the field stays required
# (the key itself must still be present), only its minimum cardinality is
# relaxed from 1 to 0.

def test_action_plan_with_empty_security_actions_parses_successfully():
    """A: ActionPlan alone, with security_actions=[], must parse cleanly."""
    from core.ai import ActionPlan

    plan = ActionPlan.model_validate(_action_plan(security_actions=[]))
    assert plan.security_actions == []


def test_committee_result_with_empty_security_actions_validates_successfully():
    """B: the full CommitteeResult (all six specialists + chairman fields)
    must validate with an empty security_actions list."""
    plan = _action_plan(security_actions=[])
    parsed = CommitteeResult.model_validate({**VALID, "action_plan": plan})
    assert parsed.action_plan.security_actions == []


def test_run_ai_analysis_accepts_no_action_portfolio_with_empty_security_actions(tmp_path, low_confidence_result):
    """C: the full run_ai_analysis path -- including _validate_action_plan_
    facts/_limits/_quality/_risk_confidence_language -- must accept a
    genuinely no-action response end-to-end, not just at the bare-model
    level."""
    plan = _action_plan(security_actions=[])
    parsed, meta = run_ai_analysis(low_confidence_result, provider=FakeProvider({**VALID, "action_plan": plan}), cache_dir=tmp_path)
    assert parsed.action_plan.security_actions == []
    assert meta["success"]


def test_action_plan_required_field_still_enforced_when_empty_list_allowed():
    """The key itself must still be required -- only its minimum cardinality
    was relaxed, not its presence. Mirrors test_action_plan_required_fields_
    are_enforced for this one field specifically, post-change."""
    broken = _action_plan()
    del broken["security_actions"]
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate({**VALID, "action_plan": broken})


def test_action_plan_non_empty_security_actions_behavior_unchanged():
    """F: existing non-empty security_actions behavior (parsing, ticker
    fields, priority) is untouched by relaxing the cardinality floor."""
    parsed = CommitteeResult.model_validate(VALID)
    assert len(parsed.action_plan.security_actions) == 1
    assert parsed.action_plan.security_actions[0].ticker == "NVDA"


def test_old_cached_payload_with_one_security_action_still_valid_after_cardinality_relaxation():
    """I: an old cached payload (from before this relaxation) always had
    security_actions length >= 1 -- relaxing the floor from 1 to 0 can only
    ever accept a strict superset of previously-valid payloads, so any old
    cached payload must still validate identically."""
    parsed = CommitteeResult.model_validate(VALID)
    assert len(parsed.action_plan.security_actions) >= 1


def test_no_factor_effective_n_field_added():
    """Part 7 / Part 8 F: guard against a factor-adjusted Effective N being
    smuggled into either the deterministic result model or the AI-facing
    output schema."""
    import dataclasses

    from core.models import AnalyticsResult

    field_names = {f.name for f in dataclasses.fields(AnalyticsResult)}
    assert not any("factor" in name.lower() for name in field_names)
    schema_text = json.dumps(CommitteeResult.model_json_schema(), ensure_ascii=False).lower()
    assert "factor" not in schema_text


def test_concentration_packet_keeps_direct_and_lookthrough_effective_n_as_distinct_keys(mixed_result):
    """E: the two Effective N figures are computed independently and must
    remain separate keys with (generally) different values -- mixed_result's
    VOO look-through genuinely redistributes weight across many more
    underlying names than its single direct holding."""
    from core.analytics import canonical_fact_packet

    packet = canonical_fact_packet(mixed_result)
    assert "direct_effective_n" in packet["concentration"]
    assert "lookthrough_effective_n" in packet["concentration"]
    assert packet["concentration"]["direct_effective_n"] != packet["concentration"]["lookthrough_effective_n"]


def test_fact_packet_never_contains_analyst_view_data(mixed_result):
    """Step 2A.10 Analyst View isolation: Step 2A.9's Analyst Consensus and
    12-Month Target data (core.analyst_view) is Page 1 informational-only
    and must never reach the AI fact packet -- no key at any depth may
    mention analyst consensus/target data, and core.ai/core.analytics must
    not import core.analyst_view at all."""
    import json
    from pathlib import Path

    from core.analytics import canonical_fact_packet

    packet = canonical_fact_packet(mixed_result)
    serialized = json.dumps(packet, ensure_ascii=False).lower()
    for forbidden in ("analyst", "recommendation", "targethigh", "targetlow", "targetmean", "consensus"):
        assert forbidden not in serialized
    project_root = Path(__file__).resolve().parents[1]
    for module in ("core/ai.py", "core/analytics.py"):
        source = (project_root / module).read_text(encoding="utf-8")
        assert "analyst_view" not in source
        assert "AnalystView" not in source
        assert "AnalystConsensus" not in source
        assert "AnalystTarget" not in source


# ============================================================================
# Step 2A.8 -- Final Semantic Hardening (Part A)
#
# Three production-review findings, none conditioned on look-through
# coverage (unlike Step 2A.3's P0-1/P0-2 guardrail): (A1) a current
# allocation FACT (current SGOV/VOO/AAPL/other weight) promoted into a
# "maintain current allocation" recommendation with no target-allocation
# optimizer behind it; (A2) a destination asset claimed as "low correlation"
# with no correlation model; (A3) a direct reinvestment-destination
# recommendation with no post-trade destination simulation. Plus one
# additional guardrail: no target/optimal Direct Effective N. See
# core.ai._validate_semantic_overreach_language for the one new fail-closed
# validator this step adds; everything else is prompt-only.
# ============================================================================

def test_validate_semantic_overreach_fails_closed_on_maintain_allocation_claim():
    """A1: a current-allocation fact must never become a blanket
    "maintain current allocation" recommendation."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(majority_view="维持整体资产配置框架，保持整体框架稳定。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_retain_framework_wording_variant():
    """Live production regression (Step 2A.8 manual acceptance run against
    the concentrated VOO/NVDA/SGOV/AAPL/XLK demo portfolio): a real Gemini
    response phrased the same "keep current framework" overreach as
    "保留核心资产配置框架" (retain) rather than "维持"/"保持" -- must fail
    closed the same way."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(
        _committee(chairman_decision="决定在保留核心资产配置框架的同时，针对已触发集中度警报的 NVDA 启动主动风险管控。")
    )
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_generic_keep_unchanged_filler():
    """A1: the retired "其余核心仓位暂维持不变" closing-line filler (which named
    VOO/SGOV/AAPL as examples in the production regression) must fail
    closed wherever it appears, e.g. in the checklist."""
    from core.ai import _validate_semantic_overreach_language

    payload = _committee()
    payload["action_plan"]["checklist"] = ["复核NVDA仓位", "其余核心仓位（如VOO、SGOV、AAPL）暂维持不变"]
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_sgov_current_weight_as_target():
    """A1: current SGOV weight alone must never be promoted into a
    "current allocation is optimal" / "defensive cornerstone" claim."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(main_concern="当前SGOV配置为理想平衡，属于组合的稳定基石。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


# ============================================================================
# Step 2A.10.1 -- Final no-action semantic guardrail. A live CAD-scenario
# production regression: "no NEW deterministic risk flag this round" /
# "look-through coverage is limited" was misread as license to endorse the
# current overall allocation ("保持既有持仓框架"/"保持现有直接持仓结构不变"/
# "维持配置"), the same A1 maintain-allocation overreach as Step 2A.8 but
# using a different phrase family (naming "持仓框架"/"持仓结构"/bare "配置"
# directly, not "资产配置框架"). Same guardrail (_validate_semantic_
# overreach_language, always runs regardless of coverage), extended
# vocabulary -- see _MAINTAIN_ALLOCATION_PHRASES.
# ============================================================================

def test_validate_semantic_overreach_fails_closed_on_existing_holdings_framework_claim():
    """CAD-scenario production regression, verbatim: "保持既有持仓框架" is the
    same maintain-allocation overreach as "维持整体资产配置框架", just phrased
    without the word "资产"."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(chairman_decision="保持既有持仓框架，静待后续数据。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_direct_holding_structure_unchanged_claim():
    """CAD-scenario production regression, verbatim: "保持现有直接持仓结构不变"
    appeared in Do Now, asserting the current direct-holding structure is
    correct with no independent target-allocation optimizer behind it."""
    from core.ai import _validate_semantic_overreach_language

    payload = _committee()
    payload["action_plan"]["do_now"] = ["保持现有直接持仓结构不变"]
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_bare_maintain_configuration_claim():
    """CAD-scenario production regression, verbatim: bare "维持配置" (no
    "整体"/"框架" qualifier) used as a free-text conclusion must still fail
    closed -- not only when it appears as the (now-removed) stance enum
    value, see test_committee_member_stance_no_longer_allows_maintain_
    configuration below for that separate schema-level guardrail."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(majority_view="鉴于本轮未触发新的确定性风险警报，维持配置。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_keep_existing_configuration_claim():
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(main_concern="当前证据支持保持现有配置。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_maintain_existing_holdings_claim():
    from core.ai import _validate_semantic_overreach_language

    payload = _committee()
    for member in payload["members"]:
        if member["role"] == "action_rebalancing":
            member["conclusion"] = "建议维持现有持仓，无需任何调整。"
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match=r"members\[action_rebalancing\]"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_current_configuration_reasonable_claim():
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(chairman_decision="综合评估后认为当前配置合理。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_current_structure_appropriate_claim():
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(chairman_decision="当前结构适宜，无需采取进一步行动。"))
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_no_adjustment_needed_claim():
    from core.ai import _validate_semantic_overreach_language

    payload = _committee()
    payload["action_plan"]["top_actions"] = ["无需调整现有配置"]
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="maintain-current-allocation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_allows_approved_no_action_wording():
    """The Step 2A.10.1 approved replacement framing -- "no current trade
    triggered by verified evidence," never "maintain the allocation" --
    must pass cleanly across every field it could appear in."""
    from core.ai import _validate_semantic_overreach_language

    payload = _committee(
        chairman_decision="当前证据既不足以支持大幅调仓，也不足以确认现有配置应维持不变，现阶段不触发具体交易。",
        majority_view="当前已验证信息不足以支持立即调整。",
        main_concern="等待更多可验证数据或触发条件后重新评估。",
    )
    payload["action_plan"]["do_now"] = ["核对应税账户中相关持仓的实际成本基础", "对相关持仓设定后续重新评估的触发条件"]
    committee = CommitteeResult.model_validate(payload)
    _validate_semantic_overreach_language(committee)  # must not raise


def test_validate_semantic_overreach_allows_negated_maintain_configuration_phrasing():
    """A negated use of the new bare "维持配置"/"保持现有配置" phrases is the
    correct, required framing (explicitly denying an allocation endorsement)
    and must never be rejected -- mirrors test_validate_semantic_overreach_
    allows_negated_phrasing for the Step 2A.8 phrase family."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(
        _committee(main_concern="当前证据不足以确认应维持配置，也不足以确认应保持现有配置，需等待更多数据。")
    )
    _validate_semantic_overreach_language(committee)  # must not raise


# --- Specialist verdict schema: "维持配置" removed from stance enum ---------

def test_committee_member_stance_no_longer_allows_maintain_configuration():
    """Step 2A.10.1: "维持配置" is removed from CommitteeMember.stance's
    Literal enum at the schema level -- the model can no longer structurally
    select it as a specialist verdict LABEL to endorse the current overall
    allocation (the exact production regression: "Specialist labels/content
    included: 维持配置")."""
    from core.ai import CommitteeMember

    with pytest.raises(ValidationError):
        CommitteeMember.model_validate({"role": "risk", "stance": "维持配置", "conclusion": "test"})


def test_committee_member_stance_still_allows_defer_action():
    """"暂缓行动" (defer action) already covers the correct "no current
    trigger, not an allocation endorsement" semantic and must remain a
    valid stance -- no replacement value was needed for the removed
    "维持配置"."""
    from core.ai import CommitteeMember

    member = CommitteeMember.model_validate({"role": "risk", "stance": "暂缓行动", "conclusion": "test"})
    assert member.stance == "暂缓行动"


def test_committee_member_stance_allows_all_remaining_four_values():
    from core.ai import CommitteeMember

    for stance in ("持有并优化", "降低风险", "增加风险", "暂缓行动"):
        member = CommitteeMember.model_validate({"role": "risk", "stance": stance, "conclusion": "test"})
        assert member.stance == stance


def test_committee_result_rejects_maintain_configuration_stance_anywhere_in_members():
    """The schema-level rejection holds for any of the six specialist slots,
    not only when constructing a bare CommitteeMember in isolation."""
    payload = _committee()
    payload["members"][0]["stance"] = "维持配置"
    with pytest.raises(ValidationError):
        CommitteeResult.model_validate(payload)


def test_validate_semantic_overreach_fails_closed_on_low_correlation_claim():
    """A2: no correlation model exists -- a destination asset must never be
    claimed as low/negatively correlated."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(
        _committee(chairman_decision="逐步将再平衡资金引导至低相关性资产，以降低组合风险。")
    )
    with pytest.raises(ValueError, match="correlation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_high_correlation_claim():
    """Live production regression (Step 2A.8 manual acceptance run): a real
    Gemini response cautioned against "其他高相关性科技资产" as a reinvestment
    destination -- an unverified HIGH-correlation claim is just as
    unsupported as the LOW-correlation claim the task calls out, since no
    correlation model exists in either direction."""
    from core.ai import _validate_semantic_overreach_language

    payload = _committee()
    payload["action_plan"]["do_not_now"] = ["切勿将卖出所得直接再投资于其他高相关性科技资产"]
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="correlation"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_direct_destination_recommendation():
    """A3: no post-trade destination simulation exists -- proceeds must
    never be directed at a specific or generically-typed destination."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(
        _committee(majority_view="优先将资金转向经穿透验证后确认能够降低集中度的资产。")
    )
    with pytest.raises(ValueError, match="destination"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_target_effective_n():
    """Additional Part A guardrail: Direct Effective N is descriptive only
    -- never a target this product computes or recommends reaching."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(
        _committee(main_concern="建议将直接持仓 Effective N 提升至更稳健水平。")
    )
    with pytest.raises(ValueError, match="Effective N"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_allows_negated_phrasing():
    """A negated use of the same words is the correct, required framing and
    must never be rejected."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(
        _committee(majority_view="不能将资金转向未经验证的资产，也不代表当前配置最优。")
    )
    _validate_semantic_overreach_language(committee)  # must not raise


def test_validate_semantic_overreach_allows_concentration_only_verification_wording():
    """The allowed replacement framing (concentration-only verification,
    hypothetical cash, no priority holding identified) must pass."""
    from core.ai import _validate_semantic_overreach_language

    committee = CommitteeResult.model_validate(_committee(
        majority_view="当前已验证数据未识别出需要优先处理的其他持仓。",
        main_concern="卖出所得暂存为现金，不预设再投资标的；再投资前，应验证候选资产在穿透后不会继续增加现有集中度。",
    ))
    _validate_semantic_overreach_language(committee)  # must not raise


def test_validate_semantic_overreach_scans_non_chairman_fields_too():
    """The guardrail is not chairman/majority_view-specific -- any
    specialist's conclusion making the same claim must also fail closed."""
    from core.ai import _validate_semantic_overreach_language

    payload = _committee()
    for member in payload["members"]:
        if member["role"] == "portfolio":
            member["conclusion"] = "维持整体配置框架，无需调整。"
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match=r"members\[portfolio\]"):
        _validate_semantic_overreach_language(committee)


# --- Full-path integration tests via run_ai_analysis ------------------------

def test_run_ai_analysis_fails_closed_on_maintain_allocation_production_regression(tmp_path, mixed_result):
    broken = _committee(chairman_decision="维持整体资产配置框架，保持整体框架稳定，其余核心仓位（如VOO、SGOV）暂维持不变。")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_run_ai_analysis_fails_closed_on_cad_scenario_no_action_production_regression(tmp_path, low_confidence_result):
    """Step 2A.10.1 live CAD-scenario production regression, full path: a
    portfolio with zero deterministic risk_flags and materially incomplete
    look-through coverage (low_confidence_result: Risk Level "Measured",
    XLF has no published constituent data) -- the same shape as the real
    production case (Portfolio Score 76 / Risk Level 适中 / Top 3 64.8% /
    Equity 64.2% / Bonds & Cash-like 35.4% / limited ETF look-through / no
    new deterministic risk alert). The AI must never turn "no trigger" into
    an allocation endorsement -- "保持既有持仓框架"/"维持配置"/"保持现有直接
    持仓结构不变" must all fail closed end-to-end via run_ai_analysis."""
    broken = _committee(
        chairman_decision="鉴于本轮未触发新的确定性风险警报，保持既有持仓框架，维持配置。",
    )
    broken["action_plan"]["do_now"] = ["保持现有直接持仓结构不变"]
    broken["action_plan"]["security_actions"] = []
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(low_confidence_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_run_ai_analysis_succeeds_with_cad_scenario_approved_no_action_wording(tmp_path, low_confidence_result):
    """The same CAD-scenario portfolio, phrased with the Step 2A.10.1
    approved no-action/reassess-later framing instead, must succeed
    end-to-end -- "no trigger" is a supported claim, only "maintain the
    allocation" is not."""
    ok = _committee(
        chairman_decision="当前已验证信息不足以支持立即调整，现阶段不触发具体交易，等待更多可验证数据或触发条件后重新评估。",
    )
    ok["action_plan"]["do_now"] = ["核对相关持仓的实际成本基础", "关注后续可验证的穿透数据更新"]
    ok["action_plan"]["security_actions"] = []
    parsed, meta = run_ai_analysis(low_confidence_result, provider=FakeProvider(ok), cache_dir=tmp_path)
    assert meta["success"]
    assert parsed.action_plan.security_actions == []


def test_run_ai_analysis_fails_closed_on_low_correlation_production_regression(tmp_path, mixed_result):
    broken = _committee(chairman_decision="逐步将再平衡资金引导至低相关性资产。")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_run_ai_analysis_fails_closed_on_direct_destination_production_regression(tmp_path, mixed_result):
    broken = _committee(chairman_decision="优先将资金转向经穿透验证后确认能够降低集中度的资产。")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_run_ai_analysis_fails_closed_on_target_effective_n_production_regression(tmp_path, mixed_result):
    broken = _committee(chairman_decision="将直接持仓 Effective N 提升至更稳健水平。")
    with pytest.raises(ProviderUnavailable):
        run_ai_analysis(mixed_result, provider=FakeProvider(broken), cache_dir=tmp_path)


def test_run_ai_analysis_succeeds_with_allowed_semantic_replacement_wording(tmp_path, mixed_result):
    """The allowed replacement framing must pass end-to-end: no priority
    holding identified beyond the flagged one, sale proceeds as
    hypothetical cash, concentration-only reinvestment verification."""
    ok = _committee(
        majority_view="当前已验证数据未识别出需要优先处理的其他持仓。",
        chairman_decision="卖出所得暂存为现金，不预设再投资标的；后续再投资需单独评估。",
    )
    parsed, meta = run_ai_analysis(mixed_result, provider=FakeProvider(ok), cache_dir=tmp_path)
    assert meta["success"]


def test_run_ai_analysis_still_succeeds_with_valid_nvda_reduce_on_rebound(tmp_path, mixed_result):
    """Regression: the existing valid NVDA REDUCE_ON_REBOUND fixture (VALID)
    must still pass end-to-end, unaffected by the new Step 2A.8 guardrail."""
    parsed, meta = run_ai_analysis(mixed_result, provider=FakeProvider(_committee()), cache_dir=tmp_path)
    assert meta["success"]
    assert parsed.action_plan.security_actions[0].ticker == "NVDA"
    assert parsed.action_plan.security_actions[0].action == "REDUCE_ON_REBOUND"


def test_run_ai_analysis_still_allows_empty_security_actions_under_new_guardrail(tmp_path, low_confidence_result):
    """Regression: a genuinely no-action portfolio may still return an
    empty security_actions list -- the new guardrail must not force a
    fabricated entry."""
    plan = _action_plan(security_actions=[])
    ok = {**VALID, "action_plan": plan}
    parsed, meta = run_ai_analysis(low_confidence_result, provider=FakeProvider(ok), cache_dir=tmp_path)
    assert meta["success"]
    assert parsed.action_plan.security_actions == []


def test_trade_impact_preview_still_treats_proceeds_as_hypothetical_cash(mixed_result, quote_factory):
    """Preserve the Trade Preview rule: sale proceeds become hypothetical
    cash, never an assumed reinvestment into any destination -- untouched
    by Step 2A.8 (core.trade_preview itself was not modified; this is a
    smoke check that the rule this Step 2A.8 report relies on is intact)."""
    from core.models import HoldingInput
    from core.trade_preview import build_trade_impact_preview

    holdings = [HoldingInput("NVDA", 20), HoldingInput("VOO", 50), HoldingInput("SGOV", 30)]
    quotes = quote_factory("NVDA", "VOO", "SGOV")
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=mixed_result.snapshot.cash,
        account_currency=mixed_result.snapshot.account_currency, usd_cad=1.35,
        account_type=mixed_result.snapshot.account_type, before_result=mixed_result,
        ticker="NVDA", action="REDUCE", position_reduction_pct=50.0,
    )
    assert preview is not None
    assert preview.executable_shares > 0
    # Reduced shares leave the portfolio as cash, never reallocated into
    # another holding -- NVDA's own direct weight drops and no other
    # holding's weight is touched by this function at all.
    assert preview.after_direct_weight < preview.before_direct_weight


def test_schema_bytes_match_step_2a_3_approved_baseline():
    """Step 2A.3's P0/P1 wording fixes are prompt/validation only (no schema
    change); the P1-1 follow-up makes exactly one deliberate, approved
    change -- dropping minItems=1 from security_actions (a 15-byte decrease,
    no new $defs, see _BASELINE_SCHEMA_BYTES). Step 2A.10.1 makes exactly one
    further deliberate, approved change -- dropping "维持配置" from
    CommitteeMember.stance's enum (a 16-byte decrease, no new $defs). Any
    *further* schema drift beyond that should be treated as a new finding,
    not silently rebaselined again."""
    schema = CommitteeResult.model_json_schema()
    schema_bytes = len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    assert schema_bytes == _BASELINE_SCHEMA_BYTES
    assert len(schema.get("$defs", {})) == _BASELINE_SCHEMA_DEFS
    security_actions_schema = schema["$defs"]["ActionPlan"]["properties"]["security_actions"]
    assert "minItems" not in security_actions_schema
    stance_schema = schema["$defs"]["CommitteeMember"]["properties"]["stance"]
    assert "维持配置" not in stance_schema["enum"]
    assert stance_schema["enum"] == ["持有并优化", "降低风险", "增加风险", "暂缓行动"]
    assert "security_actions" in schema["$defs"]["ActionPlan"]["required"]
