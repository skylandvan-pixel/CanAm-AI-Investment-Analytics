"""Step 2A.9 -- Analyst Consensus + 12-Month Target.

Layer 1, data-driven "what does Wall Street currently think" reference for
the Page 1 Security Profile card, inspired by (but not copying the styling
of) Google Finance's analyst summary. No LLM call anywhere in this module,
ever: Gemini/Anthropic must never touch, synthesize, or be fed this data
(see get_analyst_view's docstring) -- this is third-party market consensus
data, never CanAm's own recommendation.

V1 scope, per the Step 2A.9 audit:
  - individual stocks only (core.security_profile's existing kind_label ==
    "股票（Stock）" classification is the single source of truth -- an ETF,
    or any ticker with no profile at all, is never eligible and this module
    never makes a network call for one);
  - two independent subsections, each failing closed on its own: analyst
    consensus (buy/hold/sell counts, optionally a native Yahoo consensus
    label) via yfinance's recommendationTrend data (Ticker.recommendations),
    and the 12-month target price (high/mean/low/current) via yfinance's
    financialData module (Ticker.info) -- these are two separate Yahoo data
    modules with independently-sized analyst pools (audited live: e.g. AAPL
    shows 43 analysts in recommendationTrend vs. 38 in
    numberOfAnalystOpinions), so this module never merges their counts or
    implies one shared source.

Consensus label: the audit (NVDA/AAPL/GOOG/META, 2026-09) found Yahoo's own
`recommendationKey` (via Ticker.info) reliably present and consistent with
the recommendationTrend counts for large-cap individual stocks, so V1 maps
that native value directly (see _RECOMMENDATION_KEY_LABELS) rather than
deriving a label from thresholds. An unrecognized/missing key omits the
label and renders counts only -- never a guessed label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from core.security_profile import resolve_security_profile
from core.ticker_resolution import resolve_ticker

_RECOMMENDATION_KEY_LABELS: dict[str, str] = {
    "strong_buy": "强力买入",
    "buy": "买入",
    "hold": "持有",
    "sell": "卖出",
    "strong_sell": "强力卖出",
}


@dataclass(frozen=True)
class AnalystConsensus:
    buy: int
    hold: int
    sell: int
    total: int
    label: str | None = None  # None when no defensible native label exists


@dataclass(frozen=True)
class AnalystTarget:
    currency: str
    current: float
    high: float
    mean: float
    low: float
    high_pct: float
    mean_pct: float
    low_pct: float


@dataclass(frozen=True)
class AnalystView:
    ticker: str
    consensus: AnalystConsensus | None
    target: AnalystTarget | None


def _is_eligible_stock(ticker: str) -> bool:
    """Only individual stocks are eligible (see module docstring) -- an ETF,
    or any ticker resolve_security_profile can't classify, fails closed and
    is never eligible. Reuses the exact same canonical classification
    core.key_dates already relies on, never a second drifting definition."""
    profile = resolve_security_profile(ticker)
    return profile is not None and profile.kind_label == "股票（Stock）"


def _fetch_recommendations_default(resolved_ticker: str) -> list[dict] | None:
    import yfinance as yf

    df = yf.Ticker(resolved_ticker).recommendations
    if df is None or df.empty:
        return None
    return df.to_dict(orient="records")


def _fetch_info_default(resolved_ticker: str) -> dict | None:
    import yfinance as yf

    info = yf.Ticker(resolved_ticker).info
    return info if isinstance(info, dict) and info else None


def _build_consensus(records: list[dict] | None, recommendation_key: str | None) -> AnalystConsensus | None:
    """recommendationTrend's own most-recent snapshot ("0m") is the only row
    used -- never an older period, never a guessed/averaged blend across
    periods. Strong-buy/strong-sell are folded into the compact buy/sell
    counts (see the Step 2A.9 UI spec's 3-line count card); the label, if
    any, still reflects Yahoo's own 5-way native category. Fails closed
    (returns None) on any missing/non-numeric/negative field or an all-zero
    total -- never a fabricated 0/0/0 row."""
    if not records:
        return None
    row = next((r for r in records if r.get("period") == "0m"), None)
    if row is None:
        return None
    try:
        strong_buy = int(row["strongBuy"])
        buy_raw = int(row["buy"])
        hold = int(row["hold"])
        sell_raw = int(row["sell"])
        strong_sell = int(row["strongSell"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(v < 0 for v in (strong_buy, buy_raw, hold, sell_raw, strong_sell)):
        return None
    buy = strong_buy + buy_raw
    sell = sell_raw + strong_sell
    total = buy + hold + sell
    if total <= 0:
        return None
    label = _RECOMMENDATION_KEY_LABELS.get(recommendation_key or "")
    return AnalystConsensus(buy, hold, sell, total, label)


def _build_target(info: dict | None) -> AnalystTarget | None:
    """financialData's targetHigh/targetMean/targetLow + current(Price)
    define the 12-month target. Fails closed on a missing/non-numeric
    field, a non-positive or non-finite current price, a missing currency,
    or a malformed low<=mean<=high ordering -- Yahoo's own data is never
    silently reordered (see the Step 2A.9 spec's target-price rules)."""
    if not info:
        return None
    currency = info.get("currency")
    if not currency:
        return None
    try:
        current = info.get("currentPrice")
        if current is None:
            current = info.get("regularMarketPrice")
        current = float(current)
        high = float(info["targetHighPrice"])
        mean = float(info["targetMeanPrice"])
        low = float(info["targetLowPrice"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (current, high, mean, low)):
        return None
    if current <= 0:
        return None
    if not (low <= mean <= high):
        return None
    return AnalystTarget(
        currency, current, high, mean, low,
        (high / current - 1) * 100,
        (mean / current - 1) * 100,
        (low / current - 1) * 100,
    )


def get_analyst_view(
    ticker: str, *,
    fetch_recommendations: Callable[[str], list[dict] | None] | None = None,
    fetch_info: Callable[[str], dict | None] | None = None,
) -> AnalystView | None:
    """The single entry point the UI calls (Page 1 Security Profile card,
    below Key Dates). Never raises -- any provider error, network failure,
    or malformed shape fails closed to None (or to a None subsection) rather
    than surfacing a raw exception to the UI. Never calls Gemini/Anthropic,
    and this data is never fed into the AI Committee prompt, Portfolio
    Score, Risk Level, or Market Regime -- it is informational market
    consensus, not a CanAm recommendation.

    ETF/unclassified tickers return None before any network call is made
    (see _is_eligible_stock) -- fetch_recommendations/fetch_info are purely
    for deterministic test injection, mirroring core.key_dates's
    fetch_calendar pattern."""
    try:
        if not _is_eligible_stock(ticker):
            return None
        identity = resolve_ticker(ticker)
        fetch_rec = fetch_recommendations or _fetch_recommendations_default
        fetch_inf = fetch_info or _fetch_info_default

        try:
            records = fetch_rec(identity.resolved_ticker)
        except Exception:
            records = None

        try:
            info = fetch_inf(identity.resolved_ticker)
        except Exception:
            info = None

        recommendation_key = info.get("recommendationKey") if info else None
        consensus = _build_consensus(records, recommendation_key)
        target = _build_target(info)

        if consensus is None and target is None:
            return None
        return AnalystView(ticker, consensus, target)
    except Exception:
        # Fail closed: analyst data must never break Page 1 (Step 2A.9 spec).
        return None
