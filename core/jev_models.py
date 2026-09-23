"""Strict vendor response contract and safe optional-service result."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class NoulAnswer(Contract):
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


class RegimeProbabilities(Contract):
    BULL: float = Field(ge=0, le=1)
    NEUTRAL: float = Field(ge=0, le=1)
    BEAR: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def normalized(self):
        if not math.isclose(sum(self.model_dump().values()), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        return self


class ChoiceAnswer(Contract):
    type: Literal["choice"]
    choice: Literal["BULL", "NEUTRAL", "BEAR"]
    probabilities: RegimeProbabilities
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def winning_option(self):
        values = self.probabilities.model_dump()
        if values[self.choice] < max(values.values()) - 1e-6:
            raise ValueError("choice must have maximum probability")
        return self


class Answers(Contract):
    market_bullish: NoulAnswer
    market_regime: ChoiceAnswer


class Usage(Contract):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class JevResponse(Contract):
    model: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    answers: Answers
    usage: Usage


class JevResult(Contract):
    status: Literal["ok", "disabled", "unavailable", "invalid_response", "insufficient_data"]
    reason: str | None = None
    response: JevResponse | None = None
    latency_ms: float = 0.0
    attempts: int = 0
    diagnostic: dict | None = None
    regime_change_status: Literal["not_evaluated"] = "not_evaluated"
    regime_change_reason: Literal["insufficient_history"] = "insufficient_history"
