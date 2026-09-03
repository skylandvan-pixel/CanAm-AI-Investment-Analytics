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


class ConditionalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: Literal["now", "one_month", "three_month"]
    trigger: str = Field(min_length=1, max_length=180)
    action: str = Field(min_length=1, max_length=180)
    reason: str = Field(min_length=1, max_length=180)


class CommitteeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    members: list[CommitteeMember]
    majority_view: str = Field(min_length=1, max_length=240)
    main_concern: str = Field(min_length=1, max_length=240)
    chairman_decision: str = Field(min_length=1, max_length=320)
    actions: list[ConditionalAction]

    @field_validator("members")
    @classmethod
    def six_unique_roles(cls, value: list[CommitteeMember]):
        roles = [m.role for m in value]
        expected = {"macro", "portfolio", "risk", "valuation_data", "tax", "action_rebalancing"}
        if len(value) != 6 or set(roles) != expected:
            raise ValueError("exactly six unique specialist roles are required")
        return value

    @field_validator("actions")
    @classmethod
    def three_unique_horizons(cls, value: list[ConditionalAction]):
        horizons = [a.horizon for a in value]
        if len(value) > 3 or len(set(horizons)) != len(horizons):
            raise ValueError("at most three unique horizon actions are allowed")
        return value


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
or describe market regime as portfolio risk. Keep each specialist to one sentence. Actions must be
conditional IF→THEN decision support, never predictions, guarantees, orders, share counts or invented
numeric targets. Return exactly the JSON schema. Use Chinese for conclusions and plans.

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
    except ValidationError as exc:
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
