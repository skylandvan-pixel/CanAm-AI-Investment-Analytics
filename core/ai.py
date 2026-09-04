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


def _log_ai_failure(exc: Exception, *, provider: str, model: str, stage: str) -> None:
    """Server-side only diagnostic (Streamlit Cloud captures stderr in its
    Logs panel). Never raises, never re-formats exc for the caller -- the
    caller still re-raises the original exception unchanged, so this is
    purely additive observability, not a behavior change."""
    logger.error(
        "AI provider failure: provider=%s model=%s stage=%s exception=%s message=%s "
        "ai_provider_env=%s gemini_model_env=%s gemini_key_present=%s anthropic_key_present=%s",
        provider, model, stage, type(exc).__name__, _redact(str(exc)),
        os.getenv("AI_PROVIDER", ""), os.getenv("GEMINI_MODEL", ""),
        bool(os.getenv("GEMINI_API_KEY")), bool(os.getenv("ANTHROPIC_API_KEY")),
    )


def _log_ai_retry(exc: Exception, *, provider: str, model: str, status, attempt: int, max_attempts: int) -> None:
    """Server-side only diagnostic for a single retried transient failure
    (not yet a final failure -- _log_ai_failure still fires separately if
    every attempt is exhausted). No message/details are logged here since
    a transient 5xx is expected and recoverable; only status/attempt."""
    logger.warning(
        "AI provider transient failure: provider=%s model=%s stage=generate_content "
        "status=%s attempt=%d/%d retrying=True",
        provider, model, status, attempt, max_attempts,
    )


class CommitteeMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["macro", "portfolio", "risk", "valuation_data", "tax", "action_rebalancing"]
    stance: Literal["维持配置", "持有并优化", "降低风险", "增加风险", "暂缓行动"]
    conclusion: str = Field(min_length=1, max_length=160)


_ACTION_LABELS = (
    "持有", "持有并观察", "控制新增", "分批减仓", "分批增持", "分批减持", "暂缓行动", "复核",
    "维持配置", "降低风险", "重新平衡", "回调买入", "反弹减仓", "暂不操作", "财报后再决策",
)
_PRIORITY_LEVELS = ("高", "中高", "中", "低")
_CALENDAR_IMPORTANCE = ("高", "中", "低")
# The app deliberately never sends per-share quantity/price to the AI
# provider (see core.analytics.canonical_fact_packet -- the fact packet is a
# minimized, aggregated-weight-only view), so the Action Plan can never
# express a share count: only weight percentages, relative position-size
# changes, or qualitative sizing. weight_label hard-blocks presenting a
# scenario's subjective weighting as a statistical probability.
_SCENARIO_WEIGHT_LABELS = ("主观情景权重", "策略情景，不代表统计概率")
# The app has no verified event-date source (no earnings/Fed/BoC calendar
# feed) -- this is the only value date is ever allowed to hold besides None,
# enforced in _validate_action_plan_facts so the AI can never fabricate a
# specific calendar date.
_DATE_PENDING = "日期待确认"


class SecurityAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=12)
    action: Literal[_ACTION_LABELS]
    priority: Literal[_PRIORITY_LEVELS]
    reason: str = Field(min_length=1, max_length=200)
    execution_style: str | None = Field(default=None, max_length=100)
    trigger: str | None = Field(default=None, max_length=200)
    pause_condition: str | None = Field(default=None, max_length=200)
    review_point: str | None = Field(default=None, max_length=160)
    target: str | None = Field(default=None, max_length=100)


class TimelineHorizon(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = Field(min_length=1, max_length=200)
    actions: list[str] = Field(min_length=1, max_length=4)
    watch_holdings: list[str] = Field(default_factory=list, max_length=5)
    review_trigger: str = Field(min_length=1, max_length=200)


class ActionTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    now: TimelineHorizon
    next_30_days: TimelineHorizon
    next_3_months: TimelineHorizon
    next_6_12_months: TimelineHorizon


class DecisionCalendarItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: str = Field(min_length=1, max_length=120)
    date: str | None = Field(default=None, max_length=20)
    affected_holdings: list[str] = Field(default_factory=list, max_length=5)
    importance: Literal[_CALENDAR_IMPORTANCE]
    what_to_reassess: str = Field(min_length=1, max_length=160)


class TargetStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current: str = Field(min_length=1, max_length=160)
    next_3_months: str = Field(min_length=1, max_length=160)
    next_6_12_months: str = Field(min_length=1, max_length=160)


class TargetMigrationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(min_length=1, max_length=12)
    target_3_months: str | None = Field(default=None, max_length=60)
    target_6_12_months: str | None = Field(default=None, max_length=60)
    action: str = Field(min_length=1, max_length=100)


class ScenarioAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str = Field(min_length=1, max_length=120)
    weight_label: Literal[_SCENARIO_WEIGHT_LABELS] | None = None
    affected_holdings: list[str] = Field(default_factory=list, max_length=5)
    action: str = Field(min_length=1, max_length=160)
    pause_condition: str | None = Field(default=None, max_length=160)


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_now: str = Field(min_length=1, max_length=280)
    top_changes: list[str] = Field(min_length=1, max_length=3)
    do_now: list[str] = Field(min_length=2, max_length=4)
    do_not_now: list[str] = Field(min_length=2, max_length=4)
    security_actions: list[SecurityAction] = Field(min_length=1, max_length=5)
    target_structure: TargetStructure
    target_migration: list[TargetMigrationRow] = Field(default_factory=list, max_length=5)
    timeline: ActionTimeline
    scenarios: list[ScenarioAction] = Field(default_factory=list, max_length=3)
    decision_calendar: list[DecisionCalendarItem] = Field(default_factory=list, max_length=5)
    checklist: list[str] = Field(min_length=5, max_length=8)
    no_change: list[str] = Field(default_factory=list, max_length=8)


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
    """Fail-closed guardrails that a static JSON Schema cannot express on its
    own: every ticker the Action Plan references must be one already present
    in the canonical fact packet (never an invented security), and every
    decision_calendar date must be either omitted or the fixed placeholder
    string -- this app has no verified event-date source, so any other
    value is necessarily a fabricated date."""
    allowed = _canonical_tickers(packet)
    referenced: set[str] = {a.ticker for a in plan.security_actions}
    for horizon in (plan.timeline.now, plan.timeline.next_30_days, plan.timeline.next_3_months, plan.timeline.next_6_12_months):
        referenced |= set(horizon.watch_holdings)
    for item in plan.decision_calendar:
        referenced |= set(item.affected_holdings)
    for row in plan.target_migration:
        referenced.add(row.ticker)
    for scenario in plan.scenarios:
        referenced |= set(scenario.affected_holdings)
    unknown = referenced - allowed
    if unknown:
        raise ValueError(f"action plan referenced unverified ticker(s): {sorted(unknown)}")
    for item in plan.decision_calendar:
        if item.date is not None and item.date != _DATE_PENDING:
            raise ValueError("action plan decision_calendar included a fabricated date")


class TextProvider(Protocol):
    name: str
    model: str
    def generate(self, prompt: str, schema: dict) -> str: ...


class ProviderUnavailable(RuntimeError):
    pass


# Attempt 1 = original request, attempt 2/3 = retries -- 3 total attempts max.
# Backoff is the wait *after* an attempt fails: ~2s after attempt 1, ~5s after
# attempt 2, no wait after the final attempt (it just raises).
_GEMINI_MAX_ATTEMPTS = 3
_GEMINI_RETRY_BACKOFF_SECONDS = (2, 5)


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ProviderUnavailable("GEMINI_API_KEY not configured")
        from google import genai
        self.client = genai.Client(api_key=key)

    def generate(self, prompt: str, schema: dict) -> str:
        from google.genai import types
        from google.genai import errors as genai_errors
        if self.model.startswith("gemini-3"):
            # Gemini 3.x rejects legacy sampling params (temperature/top_p/top_k)
            # and the old thinking_budget field with 400 INVALID_ARGUMENT;
            # thinking_level is the Gemini 3.x replacement.
            generation_kwargs = {}
            thinking_config = types.ThinkingConfig(thinking_level="minimal")
        else:
            generation_kwargs = {"temperature": 0}
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(
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
        for attempt in range(1, _GEMINI_MAX_ATTEMPTS + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config,
                )
            except genai_errors.ServerError as exc:
                # ServerError is the SDK's own classification for 5xx (a
                # transient service-side failure, e.g. 503 UNAVAILABLE under
                # high demand). ClientError (4xx: bad request, auth, invalid
                # key) is a distinct subclass and is never caught here, so it
                # always fails closed immediately without retry.
                if attempt >= _GEMINI_MAX_ATTEMPTS:
                    raise
                _log_ai_retry(
                    exc, provider=self.name, model=self.model,
                    status=getattr(exc, "code", None), attempt=attempt, max_attempts=_GEMINI_MAX_ATTEMPTS,
                )
                time.sleep(_GEMINI_RETRY_BACKOFF_SECONDS[attempt - 1])
                continue
            if not getattr(response, "text", None):
                raise ProviderUnavailable("empty provider response")
            return response.text
        raise AssertionError("unreachable: loop always returns or raises")


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

action_plan is an execution roadmap, not a prediction. You never receive per-share quantity or price data
(only weights/exposures), so NEVER state a specific share count -- express sizing only as a weight-percentage
target, a relative position-size change (e.g. "减仓约当前仓位的15%-25%"), or a qualitative instruction. Prefer
concrete, executable actions/execution_style phrasing over vague ones (avoid bare "适度优化"/"关注风险"/
"适当调整"/"考虑再平衡" unless followed by a concrete rule of when/how). Fields:
- strategy_now: 1-3 concise lines naming the overall action stance and the single biggest current issue.
- top_changes: the 1-3 structural changes that actually matter this round -- prefer fewer, higher-confidence
  changes over listing something for every holding; put everything else in no_change.
- do_now / do_not_now: concrete investor actions and things to avoid right now -- never a system or
  data-maintenance task (never tell the user to "improve ETF holdings data" or similar; if ETF look-through
  coverage is incomplete per data_quality, the correct investor-facing action is to stay conservative about
  relying on it for large rebalancing, not to ask the user to fix data).
- security_actions: only the 3-5 holdings that most matter (put the rest in no_change, do not give every
  holding its own entry). ticker MUST be exactly one of the tickers already listed in top_direct_holdings or
  top_true_exposures above -- never invent a ticker or recommend a security outside this portfolio.
  execution_style describes staging/timing (e.g. "分2-3批", "回调3%-5%后启动第一批", "财报后一周复核") --
  never a share count. target must stay qualitative (e.g. "降至更合理区间") unless a numeric range is directly
  justified by the concentration/exposure/effective-n figures already in the packet, and even then express it
  as an approximate range, never spurious precision. trigger/pause_condition/review_point carry the IF/THEN
  conditional logic per action -- do not omit it, just don't make it the whole plan.
- target_structure: current/next_3_months/next_6_12_months, each one line describing the portfolio
  architecture this plan is steering toward (not a per-ticker instruction -- the overall shape).
- target_migration: only for holdings where a meaningful target actually exists (omit the rest); target_
  3_months/target_6_12_months may be a qualitative label ("维持"/"控制新增"/"待复核") or a weight-percentage
  range -- never spurious precision, never a share count. ticker MUST be one already listed above.
- timeline: exactly one entry each for now / next_30_days / next_3_months / next_6_12_months, describing
  investor actions and a re-assessment trigger for that horizon -- never an internal data/system task.
- scenarios: at most 3, only if the risk_flags/concentration/market data actually support distinct
  positioning per scenario -- omit entirely if not. weight_label may ONLY be omitted or exactly one of the two
  fixed disclaimer strings given in the schema; never state a numeric probability or percentage anywhere in a
  scenario, since these are not statistically modeled.
- decision_calendar: only portfolio-relevant events. You have no verified event-date source, so date must be
  either omitted or exactly the literal string "日期待确认" -- never a specific calendar date; any other date
  value is a fabricated fact and will be rejected.
- checklist: 5-8 concise, non-repetitive investor tasks compressing the whole plan into executable steps.
- no_change: holdings/positions that need no action this round, and the "do not overtrade" principle where
  relevant (e.g. "不因短期波动频繁换仓") -- this keeps the plan to a few high-confidence changes instead of
  manufacturing an action for every position.
If account_type is Taxable, actions and checklist should reflect tax friction (staged selling, checking cost
basis before realizing gains) without inventing a marginal tax rate or exact tax liability.

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
    try:
        raw = provider.generate(_prompt(packet), CommitteeResult.model_json_schema())
    except Exception as exc:
        _log_ai_failure(exc, provider=provider.name, model=provider.model, stage="generate_content")
        raise
    try:
        parsed = CommitteeResult.model_validate_json(raw)
        _validate_action_plan_facts(parsed.action_plan, packet)
    except (ValidationError, ValueError) as exc:
        _log_ai_failure(exc, provider=provider.name, model=provider.model, stage="response_parse")
        raise ProviderUnavailable("AI response failed schema validation") from exc
    meta = {
        "provider": provider.name, "model": provider.model, "fingerprint": fingerprint,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "success": True,
        "cache_hit": False,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"meta": meta, "result": parsed.model_dump()}, ensure_ascii=False, indent=2))
    return parsed, meta
