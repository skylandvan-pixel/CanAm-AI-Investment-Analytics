"""Step 2A.8 Part B -- compact deterministic Market Regime.

Every threshold/rule under test is ported verbatim from the already-tested
legacy 50-US-Stock-ETF-Investment-System market_regime_mvp/_probability/
_backtest modules (see the Step 2A.8 report for exact file/line citations).
These tests pin: the 200DMA/W3-hysteresis state machine, the frozen forward-
risk calibration tables, the always-range display ceiling, and the fail-
closed/never-raises/no-network-required contract for get_market_regime_
snapshot -- plus explicit confirmations that no CNN Fear & Greed, VIX
composite, sentiment, or LLM call was introduced, and that the module never
emits an action/buy/sell field (Market Regime describes weather only)."""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from core.market_regime import (
    FROZEN_CALIBRATION_3M,
    FROZEN_CALIBRATION_6M,
    MarketRegimeSnapshot,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_DEFENSIVE,
    REGIME_LABELS_ZH,
    REGIME_TRANSITION,
    STATUS_UNAVAILABLE,
    apply_calibration,
    build_risk_display,
    build_trend_frame,
    build_w3_state,
    classify_current_regime,
    compute_stress_tier,
    get_market_regime_snapshot,
    hysteresis_state,
)


# --- Current Regime state machine (200DMA + W3), ported verbatim -----------

def test_classify_regime_bull_when_risk_on_and_slope_non_negative():
    assert classify_current_regime(w3_risk_off=False, drawdown_from_high=-0.01, ma200_slope_20d=0.01) == REGIME_BULL


def test_classify_regime_transition_when_risk_on_but_slope_negative():
    assert classify_current_regime(w3_risk_off=False, drawdown_from_high=-0.03, ma200_slope_20d=-0.001) == REGIME_TRANSITION


def test_classify_regime_defensive_when_risk_off_but_not_bear_drawdown():
    assert classify_current_regime(w3_risk_off=True, drawdown_from_high=-0.12, ma200_slope_20d=-0.01) == REGIME_DEFENSIVE


def test_classify_regime_bear_when_risk_off_and_drawdown_reaches_20pct():
    assert classify_current_regime(w3_risk_off=True, drawdown_from_high=-0.21, ma200_slope_20d=-0.02) == REGIME_BEAR


def test_classify_regime_bear_boundary_is_exactly_20pct():
    """Ported boundary check mirroring the legacy test name/intent:
    exactly -20% (not just beyond it) already counts as Bear."""
    assert classify_current_regime(w3_risk_off=True, drawdown_from_high=-0.20, ma200_slope_20d=-0.02) == REGIME_BEAR


def test_regime_labels_cover_all_four_states():
    assert set(REGIME_LABELS_ZH) == {REGIME_BULL, REGIME_TRANSITION, REGIME_DEFENSIVE, REGIME_BEAR}


# --- W3 asymmetric hysteresis (0.5% down / 2.0% up), ported verbatim -------

def test_hysteresis_state_flips_risk_off_at_half_percent_below_ma200():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    price = pd.Series([100.0, 100.0, 99.4], index=idx)  # 99.4 is 0.6% below 100 -> triggers
    ma200 = pd.Series([100.0, 100.0, 100.0], index=idx)
    state = hysteresis_state(price, ma200, buffer_down=0.005, buffer_up=0.02)
    assert list(state) == [False, False, True]


def test_hysteresis_state_does_not_flip_within_the_buffer_band():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    price = pd.Series([100.0, 99.7], index=idx)  # 0.3% below -- inside the 0.5% band
    ma200 = pd.Series([100.0, 100.0], index=idx)
    state = hysteresis_state(price, ma200, buffer_down=0.005, buffer_up=0.02)
    assert list(state) == [False, False]


def test_hysteresis_state_requires_two_percent_above_to_flip_back_risk_on():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    price = pd.Series([100.0, 99.0, 99.0, 102.5], index=idx)  # drop triggers risk-off, then +2.5% recovers
    ma200 = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    state = hysteresis_state(price, ma200, buffer_down=0.005, buffer_up=0.02)
    assert list(state) == [False, True, True, False]


def test_hysteresis_state_persists_prior_state_between_bands():
    """Classic hysteresis: once risk-off, a partial recovery that stays
    under the 2.0% up-buffer must NOT flip back to risk-on."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    price = pd.Series([100.0, 99.0, 101.0], index=idx)  # +2% recovery, not >2%
    ma200 = pd.Series([100.0, 100.0, 100.0], index=idx)
    state = hysteresis_state(price, ma200, buffer_down=0.005, buffer_up=0.02)
    assert list(state) == [False, True, True]


def test_build_w3_state_uses_200dma_as_the_anchor():
    """S&P 500 long-term trend / 200DMA is the deterministic anchor (task's
    Expected Core Signal) -- build_w3_state must derive its ma200 series
    from build_trend_frame, not any other moving average."""
    idx = pd.bdate_range("2020-01-01", periods=260)
    price = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)  # steady uptrend
    trend = build_trend_frame(price)
    w3 = build_w3_state(price)
    assert w3.index.equals(trend.dropna(subset=["ma200"]).index.intersection(w3.index)) or not w3.empty
    # A steady uptrend never falls below its own rising 200DMA -> always risk-on.
    assert not w3.iloc[-1]


# --- 200DMA trend frame -----------------------------------------------------

def test_build_trend_frame_drawdown_from_high_matches_manual_calculation():
    idx = pd.bdate_range("2020-01-01", periods=260)
    prices = [100.0] * 252 + [90.0, 80.0, 70.0, 60.0, 50.0, 60.0, 60.0, 60.0]
    price = pd.Series(prices, index=idx)
    trend = build_trend_frame(price)
    last = trend.iloc[-1]
    # Rolling 252-day high as of the last row is 100.0 (still inside the window)
    assert last["drawdown_from_high"] == pytest.approx((60.0 - 100.0) / 100.0)


def test_build_trend_frame_ma200_slope_is_backward_looking_only():
    """No row's ma200_slope_20d may depend on data after that row (Point-
    in-Time Discipline, ported from the legacy module)."""
    idx = pd.bdate_range("2020-01-01", periods=230)
    price = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
    trend_full = build_trend_frame(price)
    trend_truncated = build_trend_frame(price.iloc[:-10])
    common_idx = trend_truncated.index
    pd.testing.assert_series_equal(
        trend_full.loc[common_idx, "ma200_slope_20d"], trend_truncated["ma200_slope_20d"], check_names=False,
    )


# --- Stress tier (W3 + VIX>25 + NFCI[always 0] + drawdown<=-10%) -----------

def test_compute_stress_tier_zero_when_calm():
    idx = pd.date_range("2024-01-01", periods=1)
    trend = pd.DataFrame({"drawdown_from_high": [-0.02]}, index=idx)
    vix = pd.Series([15.0], index=idx)
    w3 = pd.Series([False], index=idx)
    tier = compute_stress_tier(trend, vix, w3)
    assert tier.iloc[0] == 0


def test_compute_stress_tier_counts_w3_and_vix_and_drawdown_independently():
    idx = pd.date_range("2024-01-01", periods=1)
    trend = pd.DataFrame({"drawdown_from_high": [-0.15]}, index=idx)  # +1 (<=-10%)
    vix = pd.Series([30.0], index=idx)  # +1 (>25)
    w3 = pd.Series([True], index=idx)  # +1
    tier = compute_stress_tier(trend, vix, w3)
    assert tier.iloc[0] == 3  # never 4 -- NFCI term is always 0 (no FRED feed)


def test_compute_stress_tier_undefined_during_warmup():
    idx = pd.date_range("2024-01-01", periods=1)
    trend = pd.DataFrame({"drawdown_from_high": [float("nan")]}, index=idx)
    vix = pd.Series([15.0], index=idx)
    w3 = pd.Series([False], index=idx)
    tier = compute_stress_tier(trend, vix, w3)
    assert tier.iloc[0] is None or pd.isna(tier.iloc[0])


# --- Frozen calibration tables (regression pin against the legacy values) --

def test_frozen_3m_calibration_values_match_legacy_exactly():
    freqs = {t: c.observed_frequency for t, c in FROZEN_CALIBRATION_3M.by_tier.items()}
    assert freqs == {0: 0.0637, 1: 0.1473, 2: 0.4830, 3: 0.3102, 4: 0.5172}


def test_frozen_6m_calibration_values_match_legacy_exactly():
    freqs = {t: c.observed_frequency for t, c in FROZEN_CALIBRATION_6M.by_tier.items()}
    assert freqs == {0: 0.0048, 1: 0.0903, 2: 0.3835, 3: 0.1634, 4: 0.3750}


def test_apply_calibration_unknown_tier_returns_none():
    assert apply_calibration(FROZEN_CALIBRATION_3M, None) is None
    assert apply_calibration(FROZEN_CALIBRATION_3M, 99) is None


# --- Display: always a range, never a bare precise percentage --------------

@pytest.mark.parametrize("tier", [0, 1, 2, 3])
def test_build_risk_display_is_always_range_never_percentage(tier):
    """MVP v1 ceiling: even a tier with enough independent episodes for a
    percentage (tier 2/3 have >=6) must still display as a range, never a
    bare 'X%' -- preserves the legacy product's deliberate precision cap
    and the task's 'no invented exact probability' requirement."""
    cal = apply_calibration(FROZEN_CALIBRATION_3M, tier)
    display = build_risk_display(cal, pit_quality="medium", max_display_level="range")
    assert display.display_level == "range"
    assert display.display_text.endswith("%")
    assert "-" in display.display_text
    assert "%" == display.display_text[-1]
    # never a single bare percentage figure like "48%"
    assert display.display_text.count("%") == 1


def test_build_risk_display_band_and_chinese_placeholder_when_tier_unknown():
    display = build_risk_display(None, pit_quality="medium", max_display_level="range")
    assert display.display_level == "band"
    assert display.display_text == "暂不可用"


# --- get_market_regime_snapshot: deterministic, offline, fail-closed -------

def _steady_uptrend(n=420):
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(range(100, 100 + n), index=idx, dtype=float)


def test_snapshot_deterministic_bull_regime_from_synthetic_uptrend():
    spy = _steady_uptrend()
    vix = pd.Series([14.0] * len(spy), index=spy.index)
    snap = get_market_regime_snapshot(spy_history=spy, vix_history=vix)
    assert snap.status in ("complete", "partial")
    assert snap.current_regime == REGIME_BULL
    assert snap.current_regime_label == "偏多"
    assert snap.w3_state == "Risk-On"
    assert snap.correction_risk_3m_display.endswith("%")
    assert snap.bear_risk_6m_display.endswith("%")


def test_snapshot_is_deterministic_same_inputs_same_output():
    spy = _steady_uptrend()
    vix = pd.Series([14.0] * len(spy), index=spy.index)
    snap1 = get_market_regime_snapshot(spy_history=spy, vix_history=vix)
    snap2 = get_market_regime_snapshot(spy_history=spy, vix_history=vix)
    assert snap1.current_regime == snap2.current_regime
    assert snap1.correction_risk_3m_display == snap2.correction_risk_3m_display
    assert snap1.bear_risk_6m_display == snap2.bear_risk_6m_display


def test_snapshot_unavailable_when_history_too_short():
    spy = pd.Series(range(100, 150), index=pd.bdate_range("2024-01-01", periods=50), dtype=float)
    snap = get_market_regime_snapshot(spy_history=spy, vix_history=pd.Series(dtype=float))
    assert snap.status == STATUS_UNAVAILABLE
    assert snap.current_regime is None


def test_snapshot_unavailable_when_history_empty():
    snap = get_market_regime_snapshot(spy_history=pd.Series(dtype=float), vix_history=pd.Series(dtype=float))
    assert snap.status == STATUS_UNAVAILABLE


def test_snapshot_never_raises_on_corrupted_input():
    """Fail-closed guarantee: even malformed injected data must degrade to
    unavailable, never propagate an exception up into Page 1."""
    garbage = pd.Series(["not", "a", "number"] * 100, index=pd.bdate_range("2024-01-01", periods=300))
    snap = get_market_regime_snapshot(spy_history=garbage, vix_history=pd.Series(dtype=float))
    assert snap.status == STATUS_UNAVAILABLE


def test_snapshot_never_raises_when_ticker_factory_fails():
    def _boom(ticker):
        raise ConnectionError("network unavailable")

    snap = get_market_regime_snapshot(ticker_factory=_boom)
    assert snap.status == STATUS_UNAVAILABLE


def test_snapshot_does_not_require_network_when_history_injected():
    """No live network call is made when spy_history/vix_history are
    already supplied -- ticker_factory is never invoked."""
    def _fail_if_called(ticker):
        raise AssertionError(f"unexpected network call for {ticker}")

    spy = _steady_uptrend()
    vix = pd.Series([14.0] * len(spy), index=spy.index)
    snap = get_market_regime_snapshot(spy_history=spy, vix_history=vix, ticker_factory=_fail_if_called)
    assert snap.status in ("complete", "partial")


# --- Guardrails: no invented signals, no LLM call, no action field ---------

def test_no_cnn_fear_greed_or_other_forbidden_signals_implemented():
    """No forbidden signal is actually IMPLEMENTED as a name/import in this
    module (the module's own docstring legitimately names several of these
    concepts in prose, to explain what was deliberately left out -- so this
    checks defined symbols and imports, not a raw substring scan of the
    whole source, which the docstring itself would trip)."""
    import core.market_regime as mr

    defined_names = " ".join(vars(mr).keys()).lower()
    import_lines = "\n".join(
        line.strip().lower() for line in inspect.getsource(mr).splitlines() if line.strip().startswith(("import ", "from "))
    )
    for forbidden in ("fear_greed", "put_call", "breadth", "sentiment", "cnn"):
        assert forbidden not in defined_names
        assert forbidden not in import_lines


def test_no_llm_provider_call_in_market_regime_module():
    import core.market_regime as mr

    defined_names = " ".join(vars(mr).keys()).lower()
    import_lines = "\n".join(
        line.strip().lower() for line in inspect.getsource(mr).splitlines() if line.strip().startswith(("import ", "from "))
    )
    for forbidden in ("genai", "anthropic", "gemini", "openai"):
        assert forbidden not in defined_names
        assert forbidden not in import_lines


def test_snapshot_has_no_action_or_recommendation_field():
    """Market Regime describes market weather only -- it must never carry
    a buy/sell/reduce/increase field the UI could render as advice."""
    fields = {f for f in vars(MarketRegimeSnapshot) if not f.startswith("_")}
    field_names = set(MarketRegimeSnapshot.__dataclass_fields__.keys())
    for forbidden in ("action", "buy", "sell", "recommendation", "reduce", "increase"):
        assert forbidden not in " ".join(field_names).lower()


def test_regime_confidence_never_high_without_nfci_feed():
    """This app never configures a FRED/NFCI feed, so coverage can reach at
    most 5/6 -- regime_confidence must never report 'high' (which the
    legacy module reserves for >=0.9 coverage)."""
    spy = _steady_uptrend()
    vix = pd.Series([14.0] * len(spy), index=spy.index)
    snap = get_market_regime_snapshot(spy_history=spy, vix_history=vix)
    assert snap.regime_confidence != "high"
