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


def _log_ai_fallback(*, provider: str, primary_model: str, fallback_model: str, primary_status) -> None:
    """Server-side only diagnostic: the primary model exhausted its retries
    with a confirmed 503, so a single fallback attempt against a distinct
    model is about to be made. Only model identifiers and the status code
    that triggered the fallback are logged -- never the prompt, fact
    packet, or response content."""
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
    security_actions: list[SecurityAction] = Field(min_length=1)
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


def _validate_action_plan_quality(plan: ActionPlan) -> None:
    """Fail-closed guardrail against bare vague-trigger phrasing (product
    reduction pass, step 1). A static JSON Schema/length check cannot express
    "is this trigger executable" -- this only catches the specific bare
    phrases the product review called out, so a concrete anchored trigger
    that happens to use one of these words is never rejected."""
    for action in plan.security_actions:
        trigger = action.trigger.strip()
        if trigger in _VAGUE_TRIGGER_PHRASES:
            raise ValueError(f"action plan trigger for {action.ticker} is an unqualified vague phrase: {trigger!r}")


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
                try:
                    return self._call(self.fallback_model, prompt, schema)
                except Exception as fallback_exc:
                    _log_ai_failure(
                        fallback_exc, provider=self.name, model=self.fallback_model,
                        stage="generate_content_fallback",
                        prompt_bytes=len(prompt.encode("utf-8")),
                        schema_bytes=len(json.dumps(schema, ensure_ascii=False).encode("utf-8")),
                        fallback_active=True, attempt=_GEMINI_MAX_ATTEMPTS + 1,
                    )
                    raise
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
that includes an indirect/ETF-sourced component.

Decision-confidence rule: decision confidence must never exceed data confidence. When look-through coverage
is materially incomplete (see above), do not justify a major action -- a REDUCE-type security action, or a
top_action driving a meaningful position-size change -- primarily on an incomplete true-exposure number
treated as certain; size and justify it from the verified direct holding weight (top_direct_holdings) and
risk_flags instead, and let the incomplete look-through figure only add color/caution. This does not mean no
action is ever allowed under incomplete coverage -- if the direct data alone already shows meaningful
concentration, a risk-management action grounded in that direct data is still appropriate.

Fund-destination rule: never claim that moving proceeds from a reduced holding into another named ETF or
fund meaningfully lowers that stock's or sector's concentration unless this packet's own asset_allocation or
top_true_exposures data already supports that specific destination's effect. If the destination's
look-through effect cannot be verified from this packet, do not name a specific replacement security -- use
general wording such as "转向经穿透验证后能够降低集中度的资产" instead of naming a broad ETF by ticker.

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
  new information) does not need its own entry -- fold it into the checklist's closing
  "其余核心仓位暂维持不变" line instead. Only include a holding that genuinely needs REDUCE/ADD/staged
  execution/explicit restraint/important review. ticker MUST be exactly one of the tickers already listed in
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
  after a fixed period. reason is one concise explanation.
- timeline_now / timeline_30_days / timeline_3_months / timeline_6_12_months: at most 2-3 concise investor
  actions per horizon (never an internal data/system task), absorbing any longer-term structural migration
  (e.g. shifting toward more core-ETF weight, trimming a concentrated position over time) into the horizon
  where it actually belongs instead of a separate table. A horizon with nothing genuinely new to do should
  contain one concise observational entry instead of a fabricated action -- e.g. "观察"/"暂无新增操作，等待
  触发条件"/"仅重新评估，无需新增操作" -- never restate an action already covered in an earlier horizon just
  to fill space.
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
  "其余仓位暂维持不变" instead of listing them individually.
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
