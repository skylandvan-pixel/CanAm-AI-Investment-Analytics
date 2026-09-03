from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core.analytics import canonical_fact_packet
from core.models import AnalyticsResult


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


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ProviderUnavailable("GEMINI_API_KEY not configured")
        from google import genai
        self.client = genai.Client(api_key=key)

    def generate(self, prompt: str, schema: dict) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json", response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if not getattr(response, "text", None):
            raise ProviderUnavailable("empty provider response")
        return response.text


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
    provider = provider or configured_provider()
    cache_path = cache_dir / f"{fingerprint}-{provider.name}-{provider.model}.json"
    if use_cache and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            return CommitteeResult.model_validate(payload["result"]), {**payload["meta"], "cache_hit": True}
        except (OSError, KeyError, json.JSONDecodeError, ValidationError):
            pass
    packet = canonical_fact_packet(result)
    raw = provider.generate(_prompt(packet), CommitteeResult.model_json_schema())
    try:
        parsed = CommitteeResult.model_validate_json(raw)
    except ValidationError as exc:
        raise ProviderUnavailable("AI response failed schema validation") from exc
    meta = {
        "provider": provider.name, "model": provider.model, "fingerprint": fingerprint,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "success": True,
        "cache_hit": False,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"meta": meta, "result": parsed.model_dump()}, ensure_ascii=False, indent=2))
    return parsed, meta
