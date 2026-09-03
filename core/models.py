from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ACCOUNT_TYPES = ("TFSA", "Taxable")
_TFSA_ALIASES = {"tfsa", "免税", "免税账户", "免税账户（tfsa）"}


def normalize_account_type(value: str | None) -> str:
    """Account type is a Layer 1 input fact, never a gate: unrecognized or
    missing values fail open to Taxable rather than blocking analysis."""
    candidate = (value or "").strip().lower()
    return "TFSA" if candidate in _TFSA_ALIASES else "Taxable"


@dataclass(frozen=True)
class HoldingInput:
    ticker: str
    quantity: float
    average_cost: float | None = None
    currency: str | None = None

    def normalized(self) -> "HoldingInput":
        ticker = self.ticker.strip().upper().replace(".", "-")
        currency = self.currency.strip().upper() if self.currency else None
        if not ticker:
            raise ValueError("Ticker is required")
        if self.quantity <= 0:
            raise ValueError(f"{ticker}: quantity must be positive")
        if currency is not None and currency not in {"USD", "CAD"}:
            raise ValueError(f"{ticker}: currency must be USD or CAD")
        if self.average_cost is not None and self.average_cost < 0:
            raise ValueError(f"{ticker}: average cost cannot be negative")
        return HoldingInput(ticker, float(self.quantity), self.average_cost, currency)


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float | None
    currency: str | None
    previous_close: float | None = None
    as_of: str | None = None
    status: str = "verified"


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: float
    currency: str
    price: float | None
    market_value: float | None
    weight: float | None
    daily_change: float | None = None
    status: str = "complete"
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioSnapshot:
    account_currency: str
    cash: float
    positions: tuple[Position, ...]
    total_assets: float
    cash_weight: float
    coverage_ratio: float
    as_of: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    account_type: str = "Taxable"

    @property
    def complete_positions(self) -> tuple[Position, ...]:
        return tuple(p for p in self.positions if p.status == "complete" and p.weight is not None)


@dataclass(frozen=True)
class Exposure:
    ticker: str
    direct: float
    indirect: float
    true: float


@dataclass(frozen=True)
class RiskFlag:
    key: str
    title: str
    detail: str
    severity: str = "attention"


@dataclass(frozen=True)
class AnalyticsResult:
    snapshot: PortfolioSnapshot
    portfolio_score: float
    risk_level: str
    asset_allocation: dict[str, float]
    sector_exposure: dict[str, float]
    top_direct: tuple[tuple[str, float], ...]
    true_exposures: tuple[Exposure, ...]
    top3_concentration: float
    top5_concentration: float
    direct_effective_n: float | None
    lookthrough_effective_n: float | None
    lookthrough_coverage: str
    uncovered_etfs: tuple[str, ...]
    risk_flags: tuple[RiskFlag, ...]
    summary: str
    canonical_content: dict[str, str]
    data_quality: dict[str, Any]
