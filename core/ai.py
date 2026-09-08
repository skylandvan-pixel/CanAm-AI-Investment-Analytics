from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core.analytics import canonical_fact_packet
from core.models import AnalyticsResult

logger = logging.getLogger("canam.ai")

# Exact secret env vars this app configures. Their *values* are redacted from
# logs, never their presence -- see _redact() and _log_ai_failure().
_SECRET_ENV_VARS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "CANAM_BETA_CODES")

# Generic shapes for API keys / bearer tokens, as a defense-in-depth backstop
# in case a provider SDK ever echoes a credential back inside an exception
# message we didn't anticipate.
_REDACTION_PATTERNS = tuple(re.compile(p) for p in (
    r"(?i)bearer\s+[A-Za-z0-9\-_.]{10,}",
    r"(?i)authorization\s*:\s*\S+",
    r"AIza[0-9A-Za-z_\-]{10,}",
    r"sk-ant-[A-Za-z0-9_\-]{10,}",
    r"sk-[A-Za-z0-9]{10,}",
    r"gh[po]_[A-Za-z0-9]{10,}",
    r"AKIA[0-9A-Z]{16}",
))


def _redact(text: str) -> str:
    """Best-effort redaction for server-side diagnostic logs only -- never
    shown to end users. Strips any currently-configured secret value that
    appears verbatim in the text, plus common API-key/Bearer-token shapes.
    The secret values themselves are never logged, only whether each is
    configured (see _log_ai_failure)."""
    if not text:
        return text
    redacted = text
    for var in _SECRET_ENV_VARS:
        value = os.getenv(var)
        if not value:
            continue
        for piece in value.split(","):
            piece = piece.strip()
            if piece:
                redacted = redacted.replace(piece, "[REDACTED]")
    for pattern in _REDACTION_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _log_ai_failure(
    exc: Exception, *, provider: str, model: str, stage: str,
    prompt_bytes: int | None = None, schema_bytes: int | None = None,
    packet_bytes: int | None = None, fallback_active: bool = False,
    attempt: int | None = None,
) -> None:
    """Server-side only diagnostic (Streamlit Cloud captures stderr in its
    Logs panel). Never raises, never re-formats exc for the caller -- the
    caller still re-raises the original exception unchanged, so this is
    purely additive observability, not a behavior change.

    The extra fields below exist to let a future production 400 be compared
    directly against a known-good local reproduction (see the Gemini
    structured-output compatibility investigation): the exact request
    shape/size and whether the configured model/key strings carry invisible
    whitespace, without ever logging the key, prompt, or fact-packet
    content itself -- only byte counts and boolean/length metadata."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    http_status = getattr(exc, "code", None) or getattr(exc, "status", None)
    logger.error(
        "AI provider failure: provider=%s model=%s stage=%s exception=%s message=%s "
        "ai_provider_env=%s gemini_model_env=%s gemini_key_present=%s anthropic_key_present=%s "
        "model_repr=%r model_length=%d model_has_whitespace=%s "
        "gemini_key_length=%s gemini_key_has_whitespace=%s http_status=%s "
        "prompt_bytes=%s schema_bytes=%s packet_bytes=%s fallback_active=%s attempt=%s",
        provider, model, stage, type(exc).__name__, _redact(str(exc)),
        os.getenv("AI_PROVIDER", ""), os.getenv("GEMINI_MODEL", ""),
        bool(os.getenv("GEMINI_API_KEY")), bool(os.getenv("ANTHROPIC_API_KEY")),
        model, len(model), model != model.strip(),
        len(gemini_key) if gemini_key else None,
        (gemini_key != gemini_key.strip()) if gemini_key else None,
        http_status, prompt_bytes, schema_bytes, packet_bytes, fallback_active, attempt,
    )


def _log_ai_retry(
    exc: Exception, *, provider: str, model: str, status, attempt: int, max_attempts: int,
    stage: str = "generate_content",
) -> None:
    """Server-side only diagnostic for a single retried transient failure
    (not yet a final failure -- _log_ai_failure still fires separately if
    every attempt is exhausted). No message/details are logged here since
    a transient 5xx is expected and recoverable; only status/attempt.

    `stage` defaults to "generate_content" (the primary model's retry loop,
    unchanged); the fallback retry loop passes "generate_content_fallback"
    instead, so a fallback-model retry is never misread as a primary-model
    retry in the logs."""
    logger.warning(
        "AI provider transient failure: provider=%s model=%s stage=%s "
        "status=%s attempt=%d/%d retrying=True",
        provider, model, stage, status, attempt, max_attempts,
    )


def _log_ai_fallback(*, provider: str, primary_model: str, fallback_model: str, primary_status) -> None:
    """Server-side only diagnostic: the primary model exhausted its retries
    with a confirmed 503, so a bounded fallback attempt (at most
    _GEMINI_FALLBACK_MAX_ATTEMPTS calls, see GeminiProvider.
    _generate_with_fallback_retry) against a distinct model is about to be
    made. Only model identifiers and the status code that triggered the
    fallback are logged -- never the prompt, fact packet, or response
    content."""
    logger.warning(
        "AI provider fallback: provider=%s primary_model=%s fallback_model=%s stage=generate_content "
        "primary_status=%s fallback=True",
        provider, primary_model, fallback_model, primary_status,
    )


class CommitteeMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["macro", "portfolio", "risk", "valuation_data", "tax", "action_rebalancing"]
    stance: Literal["维持配置", "持有并优化", "降低风险", "增加风险", "暂缓行动"]
    conclusion: str = Field(min_length=1, max_length=160)


_ACTION_LABELS = (
    "HOLD", "WAIT", "REDUCE", "REDUCE_ON_REBOUND", "ADD_ON_PULLBACK",
    "STAGED_BUY", "STAGED_SELL", "CONTROL_ADDITIONS", "REVIEW_AFTER_EVENT",
)
_PRIORITY_LEVELS = ("高", "中高", "中", "低")

# Product-level caps on the simplified Action Plan (see _validate_action_plan_limits).
# Deliberately enforced in Python after parsing, not as JSON Schema
# minItems/maxItems -- keeping the provider-facing schema flat and small is
# the point of this simplification, and these are business rules (how much
# a portfolio-manager execution memo should hold), not shape constraints.
_ACTION_PLAN_LIMITS: dict[str, int] = {
    "top_actions": 3,
    "security_actions": 5,
    "reassessment_triggers": 3,
    "checklist": 6,
    "timeline_now": 3,
    "timeline_30_days": 3,
    "timeline_3_months": 3,
    "timeline_6_12_months": 3,
}


class SecurityAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=12)
    action: Literal[_ACTION_LABELS]
    priority: Literal[_PRIORITY_LEVELS]
    # The app deliberately never sends per-share quantity/price to the AI
    # provider (see core.analytics.canonical_fact_packet), so target must
    # stay a weight range, a relative position-size change, or qualitative
    # wording -- never a share count or exact price.
    target: str = Field(min_length=1, max_length=100)
    trigger: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=200)
    # Optional machine-readable counterpart to `target`, for REDUCE actions
    # only. Free text like "反弹时分阶段减仓约15%-20%" cannot be parsed
    # deterministically -- this single flat, optional field is Step 2A's
    # only schema change, letting the local Trade Impact Preview (see
    # core.trade_preview) convert an approved recommendation into actual
    # whole shares client-side. Never required; still never a share count
    # or price itself (see canonical_fact_packet).
    #
    # Deliberately no gt/lt here: those serialize to exclusiveMinimum/
    # exclusiveMaximum in the JSON Schema sent to Gemini, which is outside
    # the documented structured-output subset (this project has hit real
    # Gemini 400 INVALID_ARGUMENT failures from unsupported schema keywords
    # before -- see the runtime diagnostics work). The strict (0, 100) range
    # is instead enforced locally, after parsing, in
    # _validate_action_plan_quality.
    position_reduction_pct: float | None = Field(default=None)


class ReassessmentTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_or_condition: str = Field(min_length=1, max_length=160)
    affected_holdings: list[str] = Field(default_factory=list, max_length=5)
    reassess: str = Field(min_length=1, max_length=200)


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_now: str = Field(min_length=1, max_length=280)
    top_actions: list[str] = Field(min_length=1)
    do_now: list[str] = Field(min_length=1)
    do_not_now: list[str] = Field(min_length=1)
    # Step 2A.3 P1-1 follow-up: no min_length here (unlike the sibling list
    # fields above/below) -- a genuinely no-action portfolio must be able to
    # return an empty list rather than being structurally forced to fabricate
    # a HOLD/WAIT security_action merely to satisfy schema cardinality. Still
    # a required key (no default), so the provider must always return the
    # field -- only its cardinality floor is relaxed, not its presence.
    security_actions: list[SecurityAction]
    timeline_now: list[str] = Field(min_length=1)
    timeline_30_days: list[str] = Field(min_length=1)
    timeline_3_months: list[str] = Field(min_length=1)
    timeline_6_12_months: list[str] = Field(min_length=1)
    reassessment_triggers: list[ReassessmentTrigger] = Field(min_length=1)
    checklist: list[str] = Field(min_length=1)


class CommitteeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    members: list[CommitteeMember]
    majority_view: str = Field(min_length=1, max_length=240)
    main_concern: str = Field(min_length=1, max_length=240)
    chairman_decision: str = Field(min_length=1, max_length=320)
    action_plan: ActionPlan

    @field_validator("members")
    @classmethod
    def six_unique_roles(cls, value: list[CommitteeMember]):
        roles = [m.role for m in value]
        expected = {"macro", "portfolio", "risk", "valuation_data", "tax", "action_rebalancing"}
        if len(value) != 6 or set(roles) != expected:
            raise ValueError("exactly six unique specialist roles are required")
        return value


def _canonical_tickers(packet: dict) -> set[str]:
    """The only tickers the AI actually saw weight/exposure data for -- the
    Action Plan may reference these and nothing else (never an invented
    security). Sourced entirely from the same fact packet already sent to
    the provider; no new canonical data is introduced."""
    return (
        {h["ticker"] for h in packet["top_direct_holdings"]}
        | {x["ticker"] for x in packet["top_true_exposures"]}
    )


def _validate_action_plan_facts(plan: ActionPlan, packet: dict) -> None:
    """Fail-closed guardrail that a static JSON Schema cannot express on its
    own: every ticker the Action Plan references (a security_action's own
    ticker, or a reassessment_trigger's affected_holdings) must be one
    already present in the canonical fact packet -- never an invented
    security."""
    allowed = _canonical_tickers(packet)
    referenced: set[str] = {a.ticker for a in plan.security_actions}
    for trigger in plan.reassessment_triggers:
        referenced |= set(trigger.affected_holdings)
    unknown = referenced - allowed
    if unknown:
        raise ValueError(f"action plan referenced unverified ticker(s): {sorted(unknown)}")


def _validate_action_plan_limits(plan: ActionPlan) -> None:
    """Fail-closed guardrail enforcing the product-level size caps locally
    (see _ACTION_PLAN_LIMITS) instead of via JSON Schema minItems/maxItems --
    keeps the provider-facing schema flat while still keeping the rendered
    plan to a 1-2 minute scan."""
    for field, limit in _ACTION_PLAN_LIMITS.items():
        count = len(getattr(plan, field))
        if count > limit:
            raise ValueError(f"action plan field '{field}' had {count} items, exceeding the maximum of {limit}")


# The exact unqualified vague-trigger phrases called out in the Action Plan
# reduction pass (see _prompt): each is fine as part of a concrete, anchored
# trigger, but rejected when it is the entire trigger with nothing else
# qualifying it -- not executable on its own.
_VAGUE_TRIGGER_PHRASES = frozenset({"逢高", "明显反弹", "适度减仓", "分阶段优化"})

# Actions whose sizing is a single, well-defined "reduce X% of the current
# position" (immediately for REDUCE, or contingent on `trigger` for
# REDUCE_ON_REBOUND) -- the only two labels position_reduction_pct is valid
# for (Step 2A.2). STAGED_SELL is deliberately excluded: it represents an
# open-ended multi-tranche plan, not a single quantifiable reduction, so it
# stays preview-ineligible and pct-less, same as before Step 2A.2.
_REDUCTION_PCT_REQUIRED_ACTIONS = frozenset({"REDUCE", "REDUCE_ON_REBOUND"})


def _validate_action_plan_quality(plan: ActionPlan) -> None:
    """Fail-closed guardrail against bare vague-trigger phrasing (product
    reduction pass, step 1). A static JSON Schema/length check cannot express
    "is this trigger executable" -- this only catches the specific bare
    phrases the product review called out, so a concrete anchored trigger
    that happens to use one of these words is never rejected.

    Also enforces the REDUCE / REDUCE_ON_REBOUND / position_reduction_pct
    contract (Step 2A.1, extended in Step 2A.2): both actions always carry a
    usable (0, 100) position_reduction_pct -- a target-weight range in
    `target` (e.g. "目标权重 15%-20%") is a semantically different number
    and must never substitute for it, so either action with a null pct
    fails closed here rather than silently rendering no Trade Impact
    Preview. Conversely, any other action (HOLD, WAIT, STAGED_SELL, ...)
    carrying a non-null position_reduction_pct is contradictory structured
    output (that field's only defined meaning is "current-position
    reduction, immediate or trigger-contingent") and also fails closed,
    rather than being silently ignored. This field deliberately carries no
    gt/lt Pydantic constraint (see SecurityAction) to avoid emitting
    exclusiveMinimum/exclusiveMaximum into the Gemini-facing schema, so the
    (0, 100) range itself is also enforced here."""
    for action in plan.security_actions:
        trigger = action.trigger.strip()
        if trigger in _VAGUE_TRIGGER_PHRASES:
            raise ValueError(f"action plan trigger for {action.ticker} is an unqualified vague phrase: {trigger!r}")
        pct = action.position_reduction_pct
        if action.action in _REDUCTION_PCT_REQUIRED_ACTIONS:
            if pct is None:
                raise ValueError(
                    f"action plan {action.action} action for {action.ticker} is missing position_reduction_pct"
                )
            if not (0 < pct < 100):
                raise ValueError(f"action plan position_reduction_pct for {action.ticker} out of range (0, 100): {pct!r}")
        elif pct is not None:
            raise ValueError(
                f"action plan position_reduction_pct for {action.ticker} is only valid for "
                f"action in {sorted(_REDUCTION_PCT_REQUIRED_ACTIONS)!r}, got action={action.action!r}"
            )


# Exact/near-exact absolute-safety phrases the Step 2A.3 production review found
# the AI using to turn "no detected deterministic risk flag" into "risk is
# controlled" (P0-1), including the Direct-Effective-N variant of the same
# mistake (P0-2, e.g. "...处于受控范围"). Deliberately a short, literal phrase
# list rather than a broad NLP classifier -- see _unnegated_overconfident_phrase
# for why a bare substring match on these alone would be unsafe.
_OVERCONFIDENT_SAFETY_PHRASES = ("风险受控", "整体风险完全受控", "没有风险", "整体风险较低", "处于受控范围")

# Negation/hedging markers that, immediately before a matched phrase, mean the
# sentence is making the OPPOSITE (correct, required) claim -- e.g. "不能确认
# 风险受控" must never be rejected merely for containing the substring "风险受控".
_NEGATION_MARKERS = (
    "不能", "无法", "并非", "不是", "未能", "很难", "不代表", "不可视为", "不应视为", "尚不能", "无从", "不足以",
)
_NEGATION_WINDOW_CHARS = 10


def _first_unnegated_phrase(text: str, phrases: tuple[str, ...]) -> str | None:
    """Returns the first phrase from `phrases` present in `text` that is NOT
    immediately preceded (within _NEGATION_WINDOW_CHARS) by a negation/
    hedging marker, or None if every occurrence is negated (or there are
    none). Narrow substring logic, not general Chinese NLP. Shared by
    _unnegated_overconfident_phrase (Step 2A.3) and
    _validate_semantic_overreach_language (Step 2A.8)."""
    for phrase in phrases:
        start = 0
        while (idx := text.find(phrase, start)) != -1:
            window = text[max(0, idx - _NEGATION_WINDOW_CHARS):idx]
            if not any(marker in window for marker in _NEGATION_MARKERS):
                return phrase
            start = idx + 1
    return None


def _unnegated_overconfident_phrase(text: str) -> str | None:
    """Returns the first forbidden phrase in `text` that is NOT immediately
    preceded (within _NEGATION_WINDOW_CHARS) by a negation/hedging marker, or
    None if every occurrence is negated (or there are none). Narrow substring
    logic, not general Chinese NLP -- see _OVERCONFIDENT_SAFETY_PHRASES."""
    return _first_unnegated_phrase(text, _OVERCONFIDENT_SAFETY_PHRASES)


def _lookthrough_uncertainty_is_material(packet: dict) -> bool:
    """True when the packet's own deterministic data_quality signals mean
    look-through evidence is materially incomplete -- the same threshold
    _prompt already asks the model to key its coverage-aware wording off of,
    reused here so the validator and the prompt instruction agree on what
    counts as "materially incomplete"."""
    data_quality = packet.get("data_quality", {})
    if data_quality.get("lookthrough_coverage") in ("partial", "insufficient"):
        return True
    uncovered = data_quality.get("lookthrough_uncovered_weight")
    return uncovered is not None and uncovered > 0.05


def _narrative_strings(committee: CommitteeResult) -> list[tuple[str, str]]:
    """Every free-text field a specialist, the chairman, or the action plan
    could phrase a risk/diversification conclusion in -- the surface
    _validate_risk_confidence_language scans."""
    plan = committee.action_plan
    out: list[tuple[str, str]] = [(f"members[{m.role}].conclusion", m.conclusion) for m in committee.members]
    out += [
        ("majority_view", committee.majority_view),
        ("main_concern", committee.main_concern),
        ("chairman_decision", committee.chairman_decision),
        ("action_plan.strategy_now", plan.strategy_now),
    ]
    for field in (
        "top_actions", "do_now", "do_not_now", "checklist",
        "timeline_now", "timeline_30_days", "timeline_3_months", "timeline_6_12_months",
    ):
        out += [(f"action_plan.{field}[{i}]", s) for i, s in enumerate(getattr(plan, field))]
    for action in plan.security_actions:
        out += [
            (f"security_actions[{action.ticker}].target", action.target),
            (f"security_actions[{action.ticker}].trigger", action.trigger),
            (f"security_actions[{action.ticker}].reason", action.reason),
        ]
    for trigger in plan.reassessment_triggers:
        out += [
            ("reassessment_triggers.event_or_condition", trigger.event_or_condition),
            ("reassessment_triggers.reassess", trigger.reassess),
        ]
    return out


def _validate_risk_confidence_language(committee: CommitteeResult, packet: dict) -> None:
    """Fail-closed guardrail for Step 2A.3 P0-1/P0-2: when the packet's own
    look-through coverage is materially incomplete, "no detected deterministic
    risk flag" must never be reported anywhere in the committee's output as
    "risk is controlled" (or the Direct-Effective-N variant of the same
    claim). Only fires when _lookthrough_uncertainty_is_material -- a
    well-covered portfolio's legitimate reporting is never second-guessed
    here, since the deterministic risk_flags already cover that case. A
    negated use of the same words (e.g. "不能确认风险受控") is the correct,
    required framing and is never rejected -- see
    _unnegated_overconfident_phrase."""
    if not _lookthrough_uncertainty_is_material(packet):
        return
    for label, text in _narrative_strings(committee):
        phrase = _unnegated_overconfident_phrase(text)
        if phrase:
            raise ValueError(
                f"{label} made an overconfident safety claim {phrase!r} despite materially incomplete "
                "look-through coverage"
            )


# ============================================================================
# Step 2A.8 -- final semantic-overreach guardrails. Three production-review
# findings, none conditioned on look-through coverage (unlike the P0-1/P0-2
# guardrail above) -- this app has no target-allocation optimizer, no
# correlation model, and no post-trade destination simulation, so these
# phrase families are unsupported by any Layer 1 data this product computes,
# always, regardless of coverage. Same narrow literal-phrase-plus-negation
# approach as _OVERCONFIDENT_SAFETY_PHRASES above, not a broad classifier.
# ============================================================================

# A1: a current allocation FACT (current SGOV/VOO/AAPL/other weight) promoted
# into a "maintain current allocation" recommendation with no independent
# target-allocation optimizer behind it.
_MAINTAIN_ALLOCATION_PHRASES = (
    "维持整体资产配置框架", "保持整体框架稳定", "维持整体配置框架", "整体配置维持不变",
    "配置框架保持不变", "整体框架保持不变", "暂维持不变", "维持现有配置框架", "维持现有比例",
    "现有配置最优", "当前配置最优", "当前配置为理想平衡", "理想平衡", "防御性基石", "稳定基石",
    # Live Gemini regression (Step 2A.8 acceptance run against the concentrated
    # VOO/NVDA/SGOV/AAPL/XLK demo portfolio): the same "keep current framework"
    # overreach resurfaced as "保留" (retain) rather than "维持"/"保持".
    "保留核心资产配置框架", "保留整体资产配置框架", "保留现有配置框架", "保留整体配置框架",
)

# A2: any correlation claim about a holding/destination -- low, negative, OR
# high/positive -- with no correlation model in this product to verify any
# of them. Includes "高相关性"/"高相关"/"正相关" (a live Gemini regression:
# "切勿将卖出所得直接再投资于其他高相关性科技资产" asserts an unverified HIGH
# correlation just as ungrounded as the LOW-correlation destination claim the
# task calls out -- the root cause ("no correlation model exists") applies
# either direction).
_CORRELATION_CLAIM_PHRASES = (
    "低相关性资产", "低相关资产", "负相关资产", "相关性较低", "低相关配置", "通过低相关",
    "高相关性", "高相关资产", "正相关资产",
)

# A3: a direct reinvestment-destination recommendation with no deterministic
# post-trade destination simulation to support it -- includes the prior
# Step-1 "allowed" replacement phrase, which Step 2A.8 now retires because it
# still implied a verified concentration-reducing destination.
_DESTINATION_RECOMMENDATION_PHRASES = (
    "转向经穿透验证后确认能够降低集中度的资产", "将资金转向", "资金转向", "将资金重新投入", "优先配置某类资产",
)

# Additional Part A guardrail: no target/optimal Direct Effective N -- it is
# a descriptive metric only, this product has no target-allocation optimizer
# computing what Effective N "should" be.
_TARGET_EFFECTIVE_N_PHRASES = (
    "提升至更稳健水平", "目标 Effective N", "目标Effective N", "达到合理 Effective N", "达到合理Effective N",
)

_SEMANTIC_OVERREACH_PHRASE_GROUPS: dict[str, tuple[str, ...]] = {
    "unsupported maintain-current-allocation claim": _MAINTAIN_ALLOCATION_PHRASES,
    "unsupported low/negative-correlation claim": _CORRELATION_CLAIM_PHRASES,
    "unsupported reinvestment-destination recommendation": _DESTINATION_RECOMMENDATION_PHRASES,
    "unsupported target Direct Effective N claim": _TARGET_EFFECTIVE_N_PHRASES,
}


def _validate_semantic_overreach_language(committee: CommitteeResult) -> None:
    """Fail-closed guardrail for Step 2A.8: none of these four phrase
    families are ever supported by this product's current Layer 1 data (no
    target-allocation optimizer, no correlation model, no post-trade
    destination simulation) -- unlike _validate_risk_confidence_language,
    this always runs, regardless of look-through coverage. A negated use
    (e.g. "不应将资金转向...") is the correct, required framing and is never
    rejected -- see _first_unnegated_phrase."""
    for label, text in _narrative_strings(committee):
        for category, phrases in _SEMANTIC_OVERREACH_PHRASE_GROUPS.items():
            phrase = _first_unnegated_phrase(text, phrases)
            if phrase:
                raise ValueError(f"{label} contains a {category}: {phrase!r}")


class TextProvider(Protocol):
    name: str
    model: str
    def generate(self, prompt: str, schema: dict) -> str: ...


class ProviderUnavailable(RuntimeError):
    pass


# Attempt 1 = original request, attempt 2/3 = retries -- 3 total attempts max.
# Backoff is the wait *after* an attempt fails: ~2s after attempt 1, ~5s after
# attempt 2, no wait after the final attempt (it just raises). Unchanged by
# the Gemini reliability patch below -- primary retry count/timing/conditions
# are exactly as before.
_GEMINI_MAX_ATTEMPTS = 3
_GEMINI_RETRY_BACKOFF_SECONDS = (2, 5)

# Gemini reliability patch: the fallback model (activated only after the
# primary exhausts _GEMINI_MAX_ATTEMPTS with a confirmed 503 -- see
# GeminiProvider.generate) previously got exactly one, non-retried attempt.
# A single transient 503 on that one fallback attempt was then indistinguishable
# from a genuinely unavailable fallback model, and failed the whole analysis.
# Attempt 1 is immediate; attempt 2 happens only if attempt 1 itself fails
# with a confirmed 503, after a fixed ~4s wait. Any other fallback exception
# (a different 5xx, a 4xx ClientError, an empty/malformed response) still
# fails closed immediately, exactly as before -- see
# GeminiProvider._generate_with_fallback_retry.
_GEMINI_FALLBACK_MAX_ATTEMPTS = 2
_GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS = 4


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ProviderUnavailable("GEMINI_API_KEY not configured")
        from google import genai
        self.client = genai.Client(api_key=key)

    def _config(self, model: str, schema: dict):
        from google.genai import types
        if model.startswith("gemini-3"):
            # Gemini 3.x rejects legacy sampling params (temperature/top_p/top_k)
            # and the old thinking_budget field with 400 INVALID_ARGUMENT;
            # thinking_level is the Gemini 3.x replacement. gemini-3.5-flash
            # (the fallback) is also a Gemini 3.x model, so it takes this
            # same branch -- no separate fallback config.
            generation_kwargs = {}
            thinking_config = types.ThinkingConfig(thinking_level="minimal")
        else:
            generation_kwargs = {"temperature": 0}
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(
            # CommitteeResult.model_json_schema() is standard JSON Schema
            # (it uses `additionalProperties`, `$defs`, etc. via Pydantic's
            # extra="forbid"). `response_schema` only accepts the legacy
            # OpenAPI-3.0 Schema subset and rejects those keywords with a
            # 400 INVALID_ARGUMENT; `response_json_schema` is the SDK's
            # JSON-Schema-compatible field for exactly this case.
            response_mime_type="application/json", response_json_schema=schema,
            thinking_config=thinking_config,
            **generation_kwargs,
        )

    def _call(self, model: str, prompt: str, schema: dict) -> str:
        response = self.client.models.generate_content(
            model=model, contents=prompt, config=self._config(model, schema),
        )
        if not getattr(response, "text", None):
            raise ProviderUnavailable("empty provider response")
        return response.text

    def generate(self, prompt: str, schema: dict) -> str:
        from google.genai import errors as genai_errors
        for attempt in range(1, _GEMINI_MAX_ATTEMPTS + 1):
            try:
                return self._call(self.model, prompt, schema)
            except genai_errors.ServerError as exc:
                # ServerError is the SDK's own classification for 5xx (a
                # transient service-side failure, e.g. 503 UNAVAILABLE under
                # high demand). ClientError (4xx: bad request, auth, invalid
                # key) is a distinct subclass and is never caught here, so it
                # always fails closed immediately without retry or fallback.
                if attempt < _GEMINI_MAX_ATTEMPTS:
                    _log_ai_retry(
                        exc, provider=self.name, model=self.model,
                        status=getattr(exc, "code", None), attempt=attempt, max_attempts=_GEMINI_MAX_ATTEMPTS,
                    )
                    time.sleep(_GEMINI_RETRY_BACKOFF_SECONDS[attempt - 1])
                    continue
                # Primary exhausted. Fall back to a distinct model only on a
                # confirmed 503 (the SDK's structured status code -- never a
                # broad ServerError/5xx match, and never a string match on
                # the message); any other 5xx preserves the pre-fallback
                # behavior of failing closed right here.
                if getattr(exc, "code", None) != 503 or self.fallback_model == self.model:
                    raise
                _log_ai_fallback(
                    provider=self.name, primary_model=self.model,
                    fallback_model=self.fallback_model, primary_status=exc.code,
                )
                return self._generate_with_fallback_retry(prompt, schema)
        raise AssertionError("unreachable: loop always returns or raises")

    def _generate_with_fallback_retry(self, prompt: str, schema: dict) -> str:
        """Bounded fallback retry (Gemini reliability patch): at most
        _GEMINI_FALLBACK_MAX_ATTEMPTS calls against self.fallback_model.
        Attempt 1 is immediate; attempt 2 happens only if attempt 1 fails
        with a confirmed 503, after a fixed ~_GEMINI_FALLBACK_RETRY_BACKOFF_
        SECONDS wait -- same model, same prompt, same schema, same config as
        attempt 1 (just calling self._call again). Any other fallback
        exception (a non-503 ServerError, a ClientError/4xx, an empty
        response wrapped as ProviderUnavailable, ...) fails closed
        immediately with no further retry, exactly like the pre-patch single
        fallback attempt did. Never triggers a second, different fallback
        model -- only this one bounded retry of the existing fallback."""
        from google.genai import errors as genai_errors

        def _log_final_fallback_failure(exc: Exception, attempt: int) -> None:
            _log_ai_failure(
                exc, provider=self.name, model=self.fallback_model,
                stage="generate_content_fallback",
                prompt_bytes=len(prompt.encode("utf-8")),
                schema_bytes=len(json.dumps(schema, ensure_ascii=False).encode("utf-8")),
                fallback_active=True, attempt=_GEMINI_MAX_ATTEMPTS + attempt,
            )

        for attempt in range(1, _GEMINI_FALLBACK_MAX_ATTEMPTS + 1):
            try:
                return self._call(self.fallback_model, prompt, schema)
            except genai_errors.ServerError as exc:
                if attempt < _GEMINI_FALLBACK_MAX_ATTEMPTS and getattr(exc, "code", None) == 503:
                    _log_ai_retry(
                        exc, provider=self.name, model=self.fallback_model,
                        status=exc.code, attempt=attempt, max_attempts=_GEMINI_FALLBACK_MAX_ATTEMPTS,
                        stage="generate_content_fallback",
                    )
                    time.sleep(_GEMINI_FALLBACK_RETRY_BACKOFF_SECONDS)
                    continue
                _log_final_fallback_failure(exc, attempt)
                raise
            except Exception as exc:
                _log_final_fallback_failure(exc, attempt)
                raise
        raise AssertionError("unreachable: fallback loop always returns or raises")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not configured")
        import anthropic
        self.client = anthropic.Anthropic(api_key=key)

    def generate(self, prompt: str, schema: dict) -> str:
        response = self.client.messages.create(
            model=self.model, max_tokens=2600, temperature=0,
            system="Return only valid JSON matching the supplied contract.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        if not text:
            raise ProviderUnavailable("empty provider response")
        return text


def portfolio_fingerprint(result: AnalyticsResult) -> str:
    canonical = json.dumps(canonical_fact_packet(result), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def configured_provider() -> TextProvider:
    name = os.getenv("AI_PROVIDER", "gemini").strip().lower()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ProviderUnavailable("AI_PROVIDER is not supported")


def _prompt(packet: dict) -> str:
    return """You are the Seven-Member Investment Committee for CanAm AI Investment Analytics.
Six specialists review one immutable deterministic fact packet; the chairman then synthesizes it.
Use only facts in the packet. Never recalculate or change a number, invent a price/probability/tax rule,
or describe market regime as portfolio risk. Keep each specialist to one sentence. Return exactly the
JSON schema. Use Chinese for conclusions and plans.

Risk-confidence wording -- applies to every field below (every specialist conclusion, majority_view,
main_concern, chairman_decision, and action_plan alike), not only the risk specialist: the ABSENCE of a
triggered deterministic risk_flags entry is NOT itself evidence that the portfolio is safe or that risk is
controlled -- it only means analyzed/verified data has not tripped a new flag. When data_quality.
lookthrough_coverage is "partial"/"insufficient", or data_quality.lookthrough_uncovered_weight is materially
above 0 (roughly >0.05), never phrase a favorable risk conclusion as "风险受控"/"整体风险完全受控"/
"没有风险"/"整体风险较低"/"处于受控范围" or any equivalent unqualified safety claim -- those claim certainty
the data does not support. Instead say plainly that no NEW deterministic flag was triggered while explicitly
noting coverage/verification is incomplete and therefore does not confirm overall risk is low, e.g.
"当前已验证数据未触发新的确定性风险警报，但穿透覆盖有限，不能据此确认整体风险较低。" This distinction --
detected risk vs. unobserved/unverified risk -- must hold for every specialist, not only the risk role, and the
chairman may never upgrade a specialist's stated uncertainty into unqualified certainty in majority_view,
main_concern, or chairman_decision. Do NOT change risk_level itself to compensate -- it stays exactly the
deterministic Layer 1 value already in the packet; only the interpretive wording around it carries this
qualification when coverage is materially incomplete.

Effective N scope -- concentration.direct_effective_n measures concentration across DIRECT portfolio holdings
only (直接持仓口径的有效持仓数量) and must always be stated with that direct-holdings scope explicit
(e.g. "直接持仓口径 Effective N 约为5.76"). By itself it must never be described or implied as overall diversification,
true diversification, factor diversification, economic diversification, "risk controlled," or "sufficiently
diversified" -- especially when look-through coverage is incomplete, since a stock and an ETF that also holds
it (or several ETFs with overlapping constituents) can overlap economically in ways direct_effective_n cannot
see. concentration.lookthrough_effective_n, when not null, measures look-through concentration instead and
must stay semantically distinct from direct_effective_n -- never conflate the two or state one using the
other's label. If lookthrough_coverage is "partial"/"insufficient" or lookthrough_effective_n is null, qualify
any look-through Effective N statement the same way top_true_exposures figures are qualified below (a
limited-coverage figure, never a complete diversification measure). Never compute, name, or imply any
additional "factor-adjusted" or correlation-adjusted Effective N -- only the two Effective N figures already
in the packet exist. Direct Effective N is a purely descriptive metric -- this product has no target-
allocation optimizer computing what Effective N "should" be, so never set or imply a target/optimal Direct
Effective N (never "提升至更稳健水平"/"目标 Effective N"/"达到合理 Effective N"). You may only say it can be
observed going forward, e.g. "后续可观察 Direct Effective N 是否改善。"

Coverage-aware look-through wording -- applies to every field below, not only action_plan: each
top_true_exposures entry is built only from named ETF constituents already known to this packet; any ETF
weight with no published holdings data, or the unnamed remainder inside a covered ETF, is simply omitted
from that stock's true-exposure number, never guessed. So every true-exposure figure is a verified LOWER
BOUND on the real look-through exposure, never an overstatement. When concentration.lookthrough_effective_n
is null, or data_quality.lookthrough_coverage is "partial"/"insufficient", or data_quality.
lookthrough_uncovered_weight is materially above 0 (roughly >0.05), phrase any true-exposure figure you
cite as an identified lower bound -- e.g. "已识别 NVDA 穿透暴露：≥23.7%（当前可验证穿透覆盖率约XX%）" -- and
never call it a complete or fully precise total in that case. Regardless of coverage label, a true-exposure
figure may be stated directly without the ≥ qualifier only when that entry's indirect contribution is 0
(i.e. it comes entirely from a direct holding, with no ETF look-through involved) -- data_quality.
lookthrough_coverage being "complete" means every held ETF has some published constituent data, never that
100% of that ETF's holdings are named, so it alone never justifies dropping the qualifier for any figure
that includes an indirect/ETF-sourced component. For this same reason, never describe the coverage STATE
itself as "完全穿透覆盖"/"100%穿透"/"完整真实暴露"/fully exhaustive look-through/complete constituent
coverage -- those claim every underlying holding is identified, which "complete" never means. Prefer
plain, user-friendly wording that states the actual identified/verified weight, e.g. "已获得穿透数据，当前
可验证底层权重覆盖约38.39%", not internal jargon like "coverage=complete".

Reference-holdings-weight semantics (Step 2A.4): data_quality.lookthrough_covered_weight and
lookthrough_uncovered_weight each describe a share of the PORTFOLIO invested in equity ETFs -- neither is a
constituent-identification coverage percentage, and neither describes how much of any individual ETF's own
holdings are named (that per-ETF caveat is covered above). lookthrough_covered_weight is the portfolio weight
held in equity ETFs for which this packet has SOME reference holdings data at all -- phrase it as, e.g.,
"有参考持仓数据的股票 ETF 占组合比例约34.43%", never as identified/exhaustive constituent coverage.
lookthrough_uncovered_weight is the portfolio weight held in equity ETFs for which this packet has NO
reference holdings data at all -- phrase it as, e.g., "暂无参考持仓数据的股票 ETF 占组合比例约8.98%". These two
figures are NEVER complements of each other or of 100%: they are disjoint slices of the equity-ETF allocation
only (direct stocks, cash, and Fixed Income holdings sit outside both) -- never compute, state, or imply
"uncovered = 100% - covered_weight", and never add the two together and present the sum as total constituent
coverage. When lookthrough_coverage is "complete", the correct semantic interpretation is
"所有持有的股票 ETF 均有部分参考持仓数据" (every held equity ETF has at least some reference holdings data) --
never "底层资产全部覆盖"/"已完全识别底层持仓"/any wording implying every constituent is identified.

Market-value-coverage semantics (Step 2A.4): data_quality.valuation_coverage is the number of entered
positions with a usable, verified market value divided by the total number of entered positions -- a
position-count pricing-coverage ratio, computed the same way regardless of position size. It is NEVER a
Forward P/E, valuation-multiple, fundamental-valuation, dividend-yield, or earnings-data coverage measure, and
no such valuation metric exists anywhere in this packet -- never invent one. Never phrase this figure as
"估值数据覆盖率"/"估值覆盖率"/"valuation data coverage" or any wording implying fundamental valuation
coverage. Prefer "持仓市值计算覆盖率" -- e.g. "持仓市值计算覆盖率100%（按持仓数量）" for a value of 1.0.

Asset-allocation wording (Step 2A.4): asset_allocation's keys ("Equity"/"Fixed Income"/"Cash"/"Other") are
Layer 1 portfolio-weight buckets, not a stock/bond dichotomy -- the "Fixed Income" bucket includes cash-like
short-duration Treasury ETFs (e.g. SGOV/CBIL), not only traditional longer-duration bonds. Never collapse it
into "债券"/"纯债券"/"传统固定收益", and never describe the portfolio with a simple "股债" (stock/bond) framing
-- use "债券与现金类" (Bonds & Cash-like) for the Fixed Income bucket instead, e.g. "股票约70.3%，债券与现金类
资产约28.7%". If asset_allocation also has a non-trivial Cash and/or Other weight worth mentioning, name it as
its own separate figure using only the packet's own values (e.g. "另有约1%现金/其他资产") -- never fold Cash
into the Fixed Income figure, and never invent or reallocate a rounding residual just to force the cited
percentages to sum to exactly 100.

Tax-lot rule: this system has no tax-lot/share-batch history and no cost-basis-currency guarantee (see
data_quality and the local Trade Impact Preview). Never claim or recommend which lot/batch of shares to
sell, a "high-cost lot," a "low-gain lot," tax-lot sequencing, or specific-identification strategy (e.g.
never "优先卖出高成本份额"). You may only say to verify actual cost basis and potential tax impact before
reducing (e.g. "减仓前核对实际成本基础与潜在税务影响").

Decision-confidence rule: decision confidence must never exceed data confidence. When look-through coverage
is materially incomplete (see above), do not justify a major action -- a REDUCE-type security action, or a
top_action driving a meaningful position-size change -- primarily on an incomplete true-exposure number
treated as certain; size and justify it from the verified direct holding weight (top_direct_holdings) and
risk_flags instead, and let the incomplete look-through figure only add color/caution. This does not mean no
action is ever allowed under incomplete coverage -- if the direct data alone already shows meaningful
concentration, a risk-management action grounded in that direct data is still appropriate.

Fund-destination rule (Step 2A.8): this product has no correlation model and no deterministic post-trade
destination simulation, so it can verify neither which asset is "low correlation" to the rest of the
portfolio nor which specific destination would actually reduce concentration after the trade. Never claim a
destination asset is "低相关性资产"/"负相关资产"/"相关性较低", and equally never claim it is "高相关性"/"正相关"
either (the same lack of a correlation model makes a HIGH-correlation caution just as unverifiable as a
LOW-correlation recommendation) -- use "未经穿透验证" instead of any correlation-direction word when cautioning
against a reinvestment destination. Never recommend moving reduce-action
proceeds into any destination, whether named by ticker (e.g. never "核心宽基ETF（如VOO）"/"如SGOV") or only by
general type (e.g. never "转向经穿透验证后确认能够降低集中度的资产"/"优先配置某类资产"/"将资金转向"/"将资金重新
投入VOO/SGOV/SPMO") -- this app has no data proving any specific or general destination actually reduces
concentration. Treat sale proceeds from any REDUCE-type action as hypothetical cash with no reinvestment
assumption, matching the local Trade Impact Preview -- e.g. "卖出所得暂存为现金，不预设再投资标的。" If
reinvestment is worth mentioning at all, limit it to a concentration-only verification step using only this
packet's own asset_allocation/top_true_exposures data, e.g. "再投资前，应验证候选资产在穿透后不会继续增加现有
集中度" / "后续再投资需单独评估。"

Current-fact-vs-target rule: any current observed value already in this packet -- current cash weight, a
current holding/ETF/sector/asset-class weight, etc. -- is a FACT about today's portfolio, never automatically
a recommendation. State it plainly as a current fact (e.g. "当前现金约0.38%") without independently
recommending that same level as a target. You may phrase a specific weight as a recommended target (e.g.
"建议现金比例保持在X%左右", "建议目标权重X%-Y%") only when you give your own independent reasoning for why
that level is appropriate given the packet's other facts (risk_flags, concentration, allocation) -- and even
then, label it explicitly as your own recommendation ("建议"/"AI建议"), never as if the packet itself defined
that target. Never restate a current percentage as "建议保持在该比例左右" merely because it happens to already
be the current value with no additional justification -- this applies to cash specifically and to every other
current weight in the packet. This product also has no target-allocation optimizer proving the current overall
allocation (across VOO/SGOV/AAPL/or any other holding) is optimal or should be maintained -- never phrase a
current allocation as "维持整体资产配置框架"/"保持整体框架稳定"/"保留核心资产配置框架"/"当前配置为理想平衡"/
"防御性基石"/"稳定基石"/"现有配置最优" ("保留"/"维持"/"保持" a current framework are all the same claim). When
no holding besides an already-flagged one needs attention this round, say so without a
maintain-current-allocation claim -- e.g. "当前已验证数据未识别出需要优先处理的其他持仓。" or "当前行动重点集中
在已触发风险警报的持仓。" -- never a blanket "其余核心仓位暂维持不变"/"其余仓位暂维持不变" line.

SGOV / Fixed-Income application of the rule above (Step 2A.4): a current SGOV holding, its current direct
weight, and the aggregate Fixed Income allocation are all facts this packet may contain -- but CURRENT WEIGHT
ALONE IS NOT SUFFICIENT EVIDENCE for a target or for a HOLD. Never state or imply, from current weight alone,
that "保持当前SGOV配置"/"维持SGOV现有比例"/"SGOV作为组合稳定基石"/"当前SGOV配置最优"/"当前固收比例应保持不变"
or any equivalent claim that the current SGOV/Fixed-Income level is optimal, a deliberate stability cornerstone,
or should remain unchanged -- Layer 1 does not compute an optimal SGOV/Fixed-Income/cash-like target and never
proves the current allocation is optimal. This does NOT ban HOLD, CONTROL_ADDITIONS, or WAIT for SGOV or any
Fixed-Income holding -- those remain legitimate whenever reasoned from packet-visible facts (e.g. risk_flags,
concentration, or the rest of asset_allocation) and clearly labeled as your own recommendation, not a Layer 1
finding; the rule only forbids treating current weight, by itself, as proof that holding is correct.

action_plan is a portfolio manager's execution memo, NOT another analysis report -- the committee members
above already explain WHY; action_plan only answers WHAT to do, WHAT to avoid, WHICH holdings matter, WHEN to
act, and WHAT would change the plan. A reader must be able to scan it in about 1-2 minutes, so stay concrete,
portfolio-specific, and short -- do not create an action for every holding, and do not repeat the committee's
reasoning. You never receive per-share quantity or price data (only weights/exposures), so NEVER state a
specific share count or exact trade price -- express sizing only as a weight-percentage target (e.g.
"目标权重 8%-10%"), a relative position-size change (e.g. "反弹时分阶段减仓约15%-20%"), or a qualitative
instruction (e.g. "维持现有仓位", "暂不追加"). Any specific numeric target or range you set is YOUR
recommendation, never a deterministically computed threshold -- phrase it as "建议目标"/"建议控制区间"/"AI
建议" rather than stating it as if the packet itself defined that number, and prefer a range over false
precision when coverage is incomplete. Prefer concrete phrasing with a when/how rule over vague filler --
avoid bare "适度优化"/"关注风险"/"保持关注"/"适当调整" unless immediately followed by a concrete action or
condition; prefer "暂不追加"/"维持"/"反弹时分阶段减仓"/"回调后分批增加"/"财报后重新评估". Fields:
- strategy_now: one concise portfolio-level line naming the overall action stance and the single biggest
  current issue.
- top_actions: at most the 1-3 most important portfolio-level actions this round -- prefer fewer,
  higher-confidence actions over listing something for every holding.
- do_now / do_not_now: concrete investor actions and things to avoid right now -- never a system or
  data-maintenance task (never tell the user to "improve ETF holdings data" or similar; if ETF look-through
  coverage is incomplete per data_quality, the correct investor-facing action is to stay conservative about
  relying on it for large rebalancing, not to ask the user to fix data).
- security_actions: prefer FEWER, higher-conviction entries -- 1-3 is the common case; do not pad up to the
  5-action maximum just to fill it. A holding that needs no real decision (already appropriately sized, no
  new information) does not need its own entry -- e.g. never manufacture an SGOV or other Fixed-Income HOLD
  card merely because that holding exists; fold it into the checklist's closing
  "当前已验证数据未识别出需要优先处理的其他持仓" line instead (never "其余核心仓位暂维持不变" -- see the
  Current-fact-vs-target rule above). Only include a holding that genuinely needs REDUCE/ADD/staged
  execution/explicit restraint/important review. If NO holding genuinely needs one of those, return
  security_actions as an EMPTY list -- an empty list is valid and preferred over a manufactured entry. Never
  fabricate a HOLD, WAIT, "maintain current weight," or "no rebalance necessary" entry merely to make this
  section non-empty; those belong in the checklist's closing maintain-unchanged line instead, never as a
  Key Security Action, unless the hold/wait decision itself is the materially important call this round (see
  below). HOLD or WAIT must not occupy priority "高" merely because the position is large or because "no
  rebalance is necessary" -- reserve "高" priority for HOLD/WAIT only when the decision to hold/wait itself is
  the materially important call this round, with a concrete reason acting now would be wrong; a routine
  no-change holding belongs in the checklist's closing line instead, not as a Key Security Action. Never
  fabricate a REDUCE, ADD, or other action merely to fill a slot or make the plan look active -- if only one
  holding genuinely warrants an entry, list only that one; if none do, list none. ticker MUST be exactly one
  of the tickers already listed in
  top_direct_holdings or top_true_exposures above -- never invent a ticker or recommend a security outside
  this portfolio. action is one of the fixed labels in the schema (e.g. HOLD, WAIT, REDUCE,
  REDUCE_ON_REBOUND, ADD_ON_PULLBACK, STAGED_BUY, STAGED_SELL, CONTROL_ADDITIONS, REVIEW_AFTER_EVENT).
  target is a weight range, relative position-size change, or qualitative instruction -- never spurious
  numeric precision, never a share count. trigger must describe an observable condition already inferable
  from this packet -- a bare, unqualified "逢高"/"明显反弹"/"适度减仓"/"分阶段优化" with no concrete anchor is
  rejected outright, so always anchor it: the position's own recent relative strength, concentration or
  true-exposure staying above the recommended range after a market move, sector exposure staying elevated
  after a rebound, a known upcoming earnings-type event, market-regime deterioration, or look-through
  coverage materially improving -- never fabricate an exact price level, percentage move, or calendar date
  not already implied by this packet. If the trigger is conditional on a market state that may never occur
  (e.g. "反弹后减仓"), also state a time-based reassessment backstop in trigger or reason (e.g. "若X个月内
  未出现该条件，应重新评估该持仓与集中度") -- this is a reassessment prompt, never an automatic forced trade
  after a fixed period. reason is one concise explanation. position_reduction_pct: REQUIRED (never null)
  whenever action is "REDUCE" or "REDUCE_ON_REBOUND" -- this holds even when target is phrased as a target
  portfolio weight, a range, or otherwise qualitative (e.g. "目标权重 15%-20%" or "分步降至更合理区间"). It
  is a completely different number from target: it is the percentage of the CURRENT SHARE POSITION to
  reduce (e.g. current position = 100 shares, position_reduction_pct = 20 => reduce by approximately 20
  shares before whole-share rounding), never a target weight, a percentage-point weight change, a
  percentage of total portfolio, or a percentage of sale proceeds. Never derive it from a target-weight
  range in target -- a target weight depends on the rest of the portfolio and future prices, not on the
  current share count, so you must independently choose a reasonable current-position reduction percentage
  as a plain number (e.g. 20) even while target still describes the target weight. For "REDUCE" this is an
  immediate proposed reduction; for "REDUCE_ON_REBOUND" it is the reduction to execute IF/WHEN trigger is
  satisfied, not now -- keep trigger explicit and never phrase target/reason as if execution were immediate
  (e.g. "如果反弹触发条件成立，届时减持约15%", never "现在立刻减持15%"). Leave position_reduction_pct null
  for every other action (e.g. STAGED_SELL is an open-ended multi-tranche plan, not one quantifiable
  reduction -- do not force a single percentage onto it).
- timeline_now / timeline_30_days / timeline_3_months / timeline_6_12_months: at most 2-3 concise investor
  actions per horizon (never an internal data/system task), absorbing any longer-term structural migration
  (e.g. shifting toward more core-ETF weight, trimming a concentrated position over time) into the horizon
  where it actually belongs instead of a separate table. A horizon with nothing genuinely new to do should
  contain one concise observational entry instead of a fabricated action -- e.g. "观察"/"暂无新增操作，等待
  触发条件"/"仅重新评估，无需新增操作" -- never restate an action already covered in an earlier horizon just
  to fill space. If several horizons genuinely have nothing new, do not repeat the same generic
  维持/观察/等待/复核 filler in all of them -- state the single governing wait condition once, in the earliest
  horizon it applies, and keep later horizons minimal rather than restating it; the goal is information
  density, not the appearance of activity.
- reassessment_triggers: at most 3 portfolio-relevant conditions that would change this plan. Each may be
  either a known calendar-type event (e.g. next earnings, next FOMC) or a state-based condition already
  inferable from this packet (e.g. concentration or true-exposure remaining/worsening beyond the recommended
  range, market-regime deterioration, look-through coverage materially improving, sector concentration
  rising further) -- do not rely on calendar events alone, and never invent a threshold number this packet
  does not already support. event_or_condition must be a description, never a specific calendar date,
  probability, or percentage weighting -- you have no verified event-date source. affected_holdings, if any,
  MUST be tickers already listed above. reassess is one concise line on what to reconsider.
- checklist: 3-6 concise, non-repetitive investor tasks compressing the whole plan into executable steps
  (never a system/data-maintenance task); if most holdings need no action, end with one compact line such as
  "当前已验证数据未识别出需要优先处理的其他持仓" or "当前行动重点集中在已触发风险警报的持仓" instead of listing
  them individually (never "其余仓位暂维持不变" -- see the Current-fact-vs-target rule above).
If account_type is Taxable, mention the resulting tax friction (staged selling, checking cost basis before
realizing gains) in at most two places total across committee members + action_plan combined -- typically
once where it materially shapes the chairman's decision or a specific security_action, and at most one
checklist item if genuinely necessary -- never restate the identical tax caveat in top_actions, do_not_now,
timeline, AND checklist all at once. Never invent a marginal tax rate or exact tax liability.

Canonical deterministic fact packet:
""" + json.dumps(packet, ensure_ascii=False, sort_keys=True)


def run_ai_analysis(
    result: AnalyticsResult, *, provider: TextProvider | None = None,
    cache_dir: Path = Path(".cache/ai"), use_cache: bool = True,
) -> tuple[CommitteeResult | None, dict]:
    """The only API-call boundary. Call this only from an explicit user action."""
    fingerprint = portfolio_fingerprint(result)
    if provider is None:
        try:
            provider = configured_provider()
        except Exception as exc:
            # Failed before we even know which provider/model was selected --
            # e.g. AI_PROVIDER unset/unsupported, or the key check inside the
            # provider's own constructor.
            _log_ai_failure(exc, provider=os.getenv("AI_PROVIDER", "?"), model="?", stage="provider_init")
            raise
    cache_path = cache_dir / f"{fingerprint}-{provider.name}-{provider.model}.json"
    if use_cache and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            return CommitteeResult.model_validate(payload["result"]), {**payload["meta"], "cache_hit": True}
        except (OSError, KeyError, json.JSONDecodeError, ValidationError):
            pass
    packet = canonical_fact_packet(result)
    prompt = _prompt(packet)
    schema = CommitteeResult.model_json_schema()
    # Byte lengths only (never the content itself) -- purely diagnostic, so a
    # future production failure can be compared against a known-good local
    # request of the same shape/size.
    packet_bytes = len(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    prompt_bytes = len(prompt.encode("utf-8"))
    schema_bytes = len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    try:
        raw = provider.generate(prompt, schema)
    except Exception as exc:
        _log_ai_failure(
            exc, provider=provider.name, model=provider.model, stage="generate_content",
            prompt_bytes=prompt_bytes, schema_bytes=schema_bytes, packet_bytes=packet_bytes,
        )
        raise
    try:
        parsed = CommitteeResult.model_validate_json(raw)
        _validate_action_plan_facts(parsed.action_plan, packet)
        _validate_action_plan_limits(parsed.action_plan)
        _validate_action_plan_quality(parsed.action_plan)
        _validate_risk_confidence_language(parsed, packet)
        _validate_semantic_overreach_language(parsed)
    except (ValidationError, ValueError) as exc:
        _log_ai_failure(
            exc, provider=provider.name, model=provider.model, stage="response_parse",
            prompt_bytes=prompt_bytes, schema_bytes=schema_bytes, packet_bytes=packet_bytes,
        )
        raise ProviderUnavailable("AI response failed schema validation") from exc
    meta = {
        "provider": provider.name, "model": provider.model, "fingerprint": fingerprint,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "success": True,
        "cache_hit": False,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"meta": meta, "result": parsed.model_dump()}, ensure_ascii=False, indent=2))
    return parsed, meta
