"""Layer 1 deterministic Market Regime (Step 2A.8).

Ported, not redesigned: every threshold, state-machine rule, and the two
frozen forward-risk calibration tables below are reused verbatim from the
already-tested legacy implementation at
50-US-Stock-ETF-Investment-System/scripts/market_regime_mvp.py +
market_regime_probability.py + market_regime_backtest.py +
stage_b2_5_robustness_research.py::hysteresis_state (see the Step 2A.8
legacy audit for exact file/line citations). Nothing here is a new macro
model -- no CNN Fear & Greed, no VIX composite, no put/call ratio, no
breadth model, no sentiment/news/economic-forecast input, no LLM call.

Deliberate difference from the legacy module: NFCI (Chicago Fed Financial
Conditions, fetched from FRED) is never wired in here, because this app has
no FRED API key infrastructure. The legacy snapshot already degrades
gracefully to an empty NFCI series when no key is supplied (its own
already-tested no-key path) -- this module simply always takes that path,
so the NFCI term of the stress tier always contributes 0. This caps the
reachable tier at 3 (of 0-4) but never changes the tier FORMULA itself, so
frozen calibration lookups stay meaningful for the tiers this app can ever
produce.

Current Regime is a deterministic classification (200DMA trend + W3
hysteresis), never a probability. The two forward-risk figures are frozen,
DEV-period-only historical bucket frequencies (Option A: empirical
historical risk buckets), always shown as a range, never a false-precision
percentage -- see MAX_DISPLAY_LEVEL below. None of this is wired into
Portfolio Score, the Action Plan, or any buy/sell trigger: it describes
market weather only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

REGIME_BULL = "Bull"
REGIME_TRANSITION = "Transition"
REGIME_DEFENSIVE = "Defensive"
REGIME_BEAR = "Bear"

# Ordinary-user Chinese labels for Page 1 (Step 2A.8 UI wording only -- the
# underlying state names/logic above are the legacy module's own).
REGIME_LABELS_ZH = {
    REGIME_BULL: "偏多",
    REGIME_TRANSITION: "中性",
    REGIME_DEFENSIVE: "风险升高",
    REGIME_BEAR: "防御",
}

# -- Reused verbatim from the legacy market_regime_mvp.py / _backtest.py ----
BEAR_DRAWDOWN_THRESHOLD = -0.20
_MA200_SLOPE_LOOKBACK = 20
_DRAWDOWN_WINDOW = 252
_W3_BUFFER_DOWN = 0.005
_W3_BUFFER_UP = 0.02
VIX_STRESS_THRESHOLD = 25.0
DRAWDOWN_STRESS_THRESHOLD = -0.10
MAX_TIER = 4

DISPLAY_PERCENTAGE = "percentage"
DISPLAY_RANGE = "range"
DISPLAY_BAND = "band"
# Same v1 ceiling the legacy MVP uses: the DEV-period tier->frequency
# relationship is not monotonic and out-of-sample discrimination is
# unstable, so a raw percentage would overstate precision -- every reading
# is capped at "range", never shown as an exact percentage.
MAX_DISPLAY_LEVEL = DISPLAY_RANGE

BAND_LOW, BAND_MODERATE, BAND_ELEVATED, BAND_HIGH = "Low", "Moderate", "Elevated", "High"

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
CONFIDENCE_INSUFFICIENT = "Insufficient"
MIN_INDEPENDENT_EPISODES_FOR_PERCENTAGE = 6
MIN_INDEPENDENT_EPISODES_FOR_RANGE = 3

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"


def _yf_ticker_factory(ticker: str):
    import yfinance as yf
    return yf.Ticker(ticker)


def fetch_price_history(ticker: str, ticker_factory=_yf_ticker_factory) -> pd.Series:
    """Full daily-close history via yfinance -- same provider this app
    already depends on (core.market), just a longer window. Fail closed:
    any provider error or empty/malformed response returns an empty
    Series, never a fabricated one. Ported verbatim from the legacy
    fetch_price_history_yf."""
    try:
        hist = ticker_factory(ticker).history(period="max", interval="1d")
    except Exception:
        return pd.Series(dtype=float)
    if hist is None or hist.empty or "Close" not in hist:
        return pd.Series(dtype=float)
    s = hist["Close"].copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index).normalize())
    return s.sort_index().dropna()


def build_trend_frame(price: pd.Series) -> pd.DataFrame:
    """Every column is `.rolling()`/`.shift()` -- backward-looking by
    construction, never uses a future row. Ported verbatim from the legacy
    market_regime_backtest.build_trend_frame."""
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    ma200 = price.rolling(200).mean()
    df["ma200"] = ma200
    ma200_prior = ma200.shift(_MA200_SLOPE_LOOKBACK)
    df["ma200_slope_20d"] = (ma200 - ma200_prior) / ma200_prior
    rolling_high = price.rolling(_DRAWDOWN_WINDOW, min_periods=_DRAWDOWN_WINDOW).max()
    df["drawdown_from_high"] = (price - rolling_high) / rolling_high
    return df


def hysteresis_state(price: pd.Series, ma200: pd.Series, buffer_down: float, buffer_up: float) -> pd.Series:
    """W3 asymmetric hysteresis: risk-off only once price falls
    buffer_down BELOW ma200; risk-on only once price rises buffer_up ABOVE
    ma200; the prior state persists in between (classic Schmitt-trigger
    construction). Ported verbatim from the legacy stage_b2_5_robustness_
    research.hysteresis_state -- thresholds are the task's own pre-declared
    values, never fitted or swept here."""
    common = price.index.intersection(ma200.index)
    p = price.reindex(common)
    m = ma200.reindex(common)
    risk_off_out = []
    state = False  # start risk-on, matches the legacy warm-up default
    for ts in common:
        pv, mv = p[ts], m[ts]
        if pd.isna(pv) or pd.isna(mv) or not mv:
            risk_off_out.append(state)
            continue
        if pv < mv * (1 - buffer_down):
            state = True
        elif pv > mv * (1 + buffer_up):
            state = False
        risk_off_out.append(state)
    return pd.Series(risk_off_out, index=common)


def build_w3_state(spy: pd.Series) -> pd.Series:
    trend = build_trend_frame(spy)
    return hysteresis_state(spy, trend["ma200"], buffer_down=_W3_BUFFER_DOWN, buffer_up=_W3_BUFFER_UP)


def classify_current_regime(
    w3_risk_off: bool, drawdown_from_high: Optional[float], ma200_slope_20d: Optional[float],
) -> str:
    """Ported verbatim from the legacy classify_current_regime:
    - Bear: W3 confirms Risk-Off AND drawdown has reached -20%.
    - Defensive: W3 confirms Risk-Off, drawdown not (yet) Bear-level.
    - Transition: W3 still Risk-On, but MA200's own slope has turned
      negative.
    - Bull: W3 Risk-On and MA200 slope non-negative."""
    if w3_risk_off and drawdown_from_high is not None and drawdown_from_high <= BEAR_DRAWDOWN_THRESHOLD:
        return REGIME_BEAR
    if w3_risk_off:
        return REGIME_DEFENSIVE
    if ma200_slope_20d is not None and ma200_slope_20d < 0:
        return REGIME_TRANSITION
    return REGIME_BULL


def _asof_value(series: pd.Series, ts) -> Optional[float]:
    if series.empty:
        return None
    v = series.asof(ts)
    return None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)


def compute_stress_tier(spy_trend: pd.DataFrame, vix: pd.Series, w3_risk_off: pd.Series) -> pd.Series:
    """0-4 composite tier, one point per stress condition -- ported
    verbatim from the legacy compute_stress_tier's four terms (W3, VIX,
    NFCI, drawdown). The NFCI term is always 0 here (see module docstring:
    no FRED key configured), so this app's tier can only ever reach 3, but
    the formula itself is unchanged -- a tier value here means the same
    thing the frozen calibration tables below were built from."""
    common = spy_trend.index.intersection(w3_risk_off.index)
    out = pd.Series(None, index=spy_trend.index, dtype="Int64")
    for ts in common:
        dd = spy_trend.at[ts, "drawdown_from_high"]
        if dd is None or (isinstance(dd, float) and pd.isna(dd)):
            continue  # warm-up / missing trend data -> tier undefined, never guessed
        vix_level = _asof_value(vix, ts)
        w3 = w3_risk_off.get(ts)
        tier = 0
        tier += 1 if bool(w3) else 0
        tier += 1 if (vix_level is not None and vix_level > VIX_STRESS_THRESHOLD) else 0
        tier += 0  # NFCI term: always 0, no FRED/NFCI feed configured
        tier += 1 if dd <= DRAWDOWN_STRESS_THRESHOLD else 0
        out[ts] = tier
    return out


@dataclass
class TierCalibration:
    tier: int
    n_observations: int
    n_independent_episodes: int
    observed_frequency: Optional[float]


@dataclass
class CalibrationTable:
    horizon_label: str
    dev_period: tuple[str, str]
    by_tier: dict[int, TierCalibration]


def apply_calibration(table: CalibrationTable, current_tier: Optional[int]) -> Optional[TierCalibration]:
    if current_tier is None or current_tier not in table.by_tier:
        return None
    return table.by_tier[current_tier]


def confidence_for_tier(cal: Optional[TierCalibration], pit_quality: str = "medium") -> str:
    if cal is None or cal.observed_frequency is None:
        return CONFIDENCE_INSUFFICIENT
    if cal.n_independent_episodes >= MIN_INDEPENDENT_EPISODES_FOR_PERCENTAGE and pit_quality in ("high", "medium"):
        return CONFIDENCE_HIGH if pit_quality == "high" else CONFIDENCE_MEDIUM
    if cal.n_independent_episodes >= MIN_INDEPENDENT_EPISODES_FOR_RANGE:
        return CONFIDENCE_LOW
    return CONFIDENCE_INSUFFICIENT


def display_level_for_confidence(confidence: str) -> str:
    if confidence in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM):
        return DISPLAY_PERCENTAGE
    if confidence == CONFIDENCE_LOW:
        return DISPLAY_RANGE
    return DISPLAY_BAND


def band_for_frequency(freq: Optional[float]) -> str:
    if freq is None:
        return BAND_LOW
    if freq < 0.10:
        return BAND_LOW
    if freq < 0.25:
        return BAND_MODERATE
    if freq < 0.45:
        return BAND_ELEVATED
    return BAND_HIGH


@dataclass
class RiskDisplay:
    display_level: str  # "percentage" / "range" / "band"
    display_text: str
    numeric_value: Optional[float]
    confidence: str


_DISPLAY_LEVEL_RANK = {DISPLAY_BAND: 0, DISPLAY_RANGE: 1, DISPLAY_PERCENTAGE: 2}


def build_risk_display(
    cal: Optional[TierCalibration], pit_quality: str = "medium", max_display_level: Optional[str] = None,
) -> RiskDisplay:
    """`max_display_level` is a ceiling, never a floor: it can only
    downgrade a level the sample-size confidence gate would otherwise
    allow (percentage -> range), it can never upgrade one. Ported verbatim
    from the legacy build_risk_display."""
    confidence = confidence_for_tier(cal, pit_quality)
    level = display_level_for_confidence(confidence)
    if max_display_level is not None and _DISPLAY_LEVEL_RANK[level] > _DISPLAY_LEVEL_RANK[max_display_level]:
        level = max_display_level
    if cal is None:
        return RiskDisplay(DISPLAY_BAND, "暂不可用", None, CONFIDENCE_INSUFFICIENT)

    freq = cal.observed_frequency
    if level == DISPLAY_PERCENTAGE and freq is not None:
        text = f"{round(freq * 100)}%"
    elif level == DISPLAY_RANGE and freq is not None:
        lo, hi = max(0.0, freq - 0.10), min(1.0, freq + 0.10)
        text = f"{round(lo * 100)}-{round(hi * 100)}%"
    else:
        text = band_for_frequency(freq)
    return RiskDisplay(display_level=level, display_text=text, numeric_value=freq, confidence=confidence)


# ===========================================================================
# FROZEN CALIBRATION TABLES -- copied verbatim from the legacy market_regime_
# mvp.py's FROZEN_CALIBRATION_3M/6M (precomputed once from 2000-2014 SPY/VIX/
# NFCI history; see that module and research_outputs/
# market_regime_mvp_validation.md in the legacy repo for the full
# derivation). Looked up here, never recomputed live.
# ===========================================================================

FROZEN_CALIBRATION_3M = CalibrationTable(
    horizon_label="3m_correction", dev_period=("2000-01-01", "2014-12-31"),
    by_tier={
        0: TierCalibration(0, n_observations=2102, n_independent_episodes=5, observed_frequency=0.0637),
        1: TierCalibration(1, n_observations=421, n_independent_episodes=5, observed_frequency=0.1473),
        2: TierCalibration(2, n_observations=412, n_independent_episodes=6, observed_frequency=0.4830),
        3: TierCalibration(3, n_observations=606, n_independent_episodes=7, observed_frequency=0.3102),
        4: TierCalibration(4, n_observations=232, n_independent_episodes=4, observed_frequency=0.5172),
    },
)
FROZEN_CALIBRATION_6M = CalibrationTable(
    horizon_label="6m_bear", dev_period=("2000-01-01", "2014-12-31"),
    by_tier={
        0: TierCalibration(0, n_observations=2102, n_independent_episodes=5, observed_frequency=0.0048),
        1: TierCalibration(1, n_observations=421, n_independent_episodes=5, observed_frequency=0.0903),
        2: TierCalibration(2, n_observations=412, n_independent_episodes=6, observed_frequency=0.3835),
        3: TierCalibration(3, n_observations=606, n_independent_episodes=7, observed_frequency=0.1634),
        4: TierCalibration(4, n_observations=232, n_independent_episodes=4, observed_frequency=0.3750),
    },
)


@dataclass
class MarketRegimeSnapshot:
    status: str = STATUS_UNAVAILABLE
    as_of: Optional[str] = None
    current_regime: Optional[str] = None
    current_regime_label: Optional[str] = None
    regime_confidence: str = "unknown"
    w3_state: Optional[str] = None  # "Risk-On" / "Risk-Off"
    price_vs_ma200: Optional[str] = None  # "above" / "below"
    correction_risk_3m_display: Optional[str] = None
    bear_risk_6m_display: Optional[str] = None
    data_coverage: Optional[float] = None


def get_market_regime_snapshot(
    spy_history: Optional[pd.Series] = None,
    vix_history: Optional[pd.Series] = None,
    ticker_factory=_yf_ticker_factory,
) -> MarketRegimeSnapshot:
    """Single entry point. NEVER raises -- any internal failure (missing/
    insufficient market data, a provider error, unexpected NaNs) degrades
    to status=STATUS_UNAVAILABLE so Page 1 can omit the module cleanly
    rather than break. `spy_history`/`vix_history` are injectable so tests
    never need live network access; production callers normally omit them
    and let this function fetch via yfinance -- the same provider
    core.market already depends on."""
    try:
        as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if spy_history is None:
            spy_history = fetch_price_history("SPY", ticker_factory)
        if vix_history is None:
            vix_history = fetch_price_history("^VIX", ticker_factory)

        if spy_history.empty or len(spy_history) < 200:
            return MarketRegimeSnapshot(status=STATUS_UNAVAILABLE, as_of=as_of)

        trend = build_trend_frame(spy_history)
        last_ts = trend.index[-1]
        price = trend.at[last_ts, "price"]
        ma200 = trend.at[last_ts, "ma200"]
        slope = trend.at[last_ts, "ma200_slope_20d"]
        drawdown = trend.at[last_ts, "drawdown_from_high"]
        if price is None or pd.isna(ma200):
            return MarketRegimeSnapshot(status=STATUS_UNAVAILABLE, as_of=as_of)

        w3_series = build_w3_state(spy_history)
        w3_risk_off = bool(w3_series.get(last_ts, False))
        slope_value = None if pd.isna(slope) else float(slope)
        drawdown_value = None if pd.isna(drawdown) else float(drawdown)
        current_regime = classify_current_regime(w3_risk_off, drawdown_value, slope_value)

        available = [price is not None, not pd.isna(ma200), slope_value is not None, drawdown_value is not None]
        vix_available = not vix_history.empty
        nfci_available = False  # no FRED/NFCI feed configured -- see module docstring
        coverage_components = available + [vix_available, nfci_available]
        coverage = sum(1 for c in coverage_components if c) / len(coverage_components)
        regime_confidence = "high" if coverage >= 0.9 else ("medium" if coverage >= 0.6 else "low")

        tier_series = compute_stress_tier(trend, vix_history, w3_series)
        current_tier = tier_series.get(last_ts)
        current_tier = int(current_tier) if current_tier is not None else None

        cal_3m = apply_calibration(FROZEN_CALIBRATION_3M, current_tier)
        cal_6m = apply_calibration(FROZEN_CALIBRATION_6M, current_tier)
        risk_3m = build_risk_display(cal_3m, pit_quality="medium", max_display_level=MAX_DISPLAY_LEVEL)
        risk_6m = build_risk_display(cal_6m, pit_quality="medium", max_display_level=MAX_DISPLAY_LEVEL)

        status = STATUS_COMPLETE if coverage >= 0.9 else STATUS_PARTIAL

        return MarketRegimeSnapshot(
            status=status, as_of=as_of,
            current_regime=current_regime, current_regime_label=REGIME_LABELS_ZH[current_regime],
            regime_confidence=regime_confidence,
            w3_state=("Risk-Off" if w3_risk_off else "Risk-On"),
            price_vs_ma200=("above" if price >= ma200 else "below"),
            correction_risk_3m_display=risk_3m.display_text,
            bear_risk_6m_display=risk_6m.display_text,
            data_coverage=round(coverage, 3),
        )
    except Exception:  # noqa: BLE001 - Market Regime failure must never break Page 1.
        return MarketRegimeSnapshot(status=STATUS_UNAVAILABLE, as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"))
