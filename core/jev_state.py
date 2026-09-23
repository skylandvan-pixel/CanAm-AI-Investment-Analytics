"""Build a market-only state using existing history/trend functions.

Conservatively exclude today's daily bar (possibly incomplete). Freshness is
calendar-day based, not an exchange-calendar claim. No fabricated indicators.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from core.market_regime import build_trend_frame, build_w3_state, classify_current_regime, fetch_price_history


def build_market_state(spy: pd.Series, vix: pd.Series, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    today = pd.Timestamp(now.astimezone(timezone.utc).date())

    def usable(series):
        series = series.copy()
        series.index = pd.DatetimeIndex(series.index)
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        series.index = series.index.normalize()
        series = series.sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series[series.index < today].dropna()

    spy = usable(spy)
    if len(spy) < 252 or (today - spy.index[-1]).days > 7:
        raise ValueError("insufficient or stale SPY data")
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in spy):
        raise ValueError("invalid SPY data")
    trend = build_trend_frame(spy).iloc[-1]
    risk_off = bool(build_w3_state(spy).iloc[-1])
    data_date = spy.index[-1]
    state = {
        "schema_version": "market-state-v1",
        "target_market": "U.S. equities (S&P 500 proxy, not the entire market)",
        "benchmark_symbol": "SPY",
        "data_timestamp": data_date.date().isoformat(),
        "timestamp_precision": "daily_bar_date",
        "generated_timestamp": now.astimezone(timezone.utc).isoformat(),
        "source": "yfinance",
        "price_basis": "existing market_regime.fetch_price_history default adjustment; not raw quote price",
        "bar_policy": "exclude current UTC date and future bars",
        "spy": {
            "close": float(trend["price"]), "ma200": float(trend["ma200"]),
            "price_vs_ma200": "above" if trend["price"] >= trend["ma200"] else "below",
            "ma200_slope_20d": float(trend["ma200_slope_20d"]),
            "drawdown_from_252d_high": float(trend["drawdown_from_high"]),
            "w3_state": "Risk-Off" if risk_off else "Risk-On",
            "existing_regime": classify_current_regime(risk_off, float(trend["drawdown_from_high"]), float(trend["ma200_slope_20d"])),
        },
        "vix": {"status": "unavailable"},
        "unavailable": ["RSI", "MA20", "MA50", "Treasury yields", "Fed policy", "valuation", "earnings", "market breadth"],
    }
    vix = usable(vix)
    vix = vix[vix.index <= data_date]
    if not vix.empty and (data_date - vix.index[-1]).days <= 7:
        value = float(vix.iloc[-1])
        if math.isfinite(value) and value > 0:
            state["vix"] = {"status": "available", "symbol": "^VIX", "value": value,
                            "data_timestamp": vix.index[-1].date().isoformat()}
    return state


def current_market_state() -> dict:
    """Only called by the explicit harness, after the service's flag/key gate."""
    return build_market_state(fetch_price_history("SPY"), fetch_price_history("^VIX"))
