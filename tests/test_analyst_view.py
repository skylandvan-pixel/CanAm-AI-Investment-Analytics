"""Step 2A.9 -- Analyst Consensus + 12-Month Target.

Deterministic, dependency-injected tests (mirrors tests/test_key_dates.py's
fetch_calendar pattern): fetch_recommendations/fetch_info are always
injected here, so this module never makes a live network call and never
depends on the real world's analyst data drifting over time.
"""

from __future__ import annotations

from core.analyst_view import get_analyst_view

_NVDA_RECORDS = [
    {"period": "0m", "strongBuy": 9, "buy": 48, "hold": 2, "sell": 1, "strongSell": 0},
    {"period": "-1m", "strongBuy": 9, "buy": 48, "hold": 2, "sell": 1, "strongSell": 0},
]

_NVDA_INFO = {
    "recommendationKey": "strong_buy",
    "currency": "USD",
    "currentPrice": 218.68,
    "targetHighPrice": 515.0,
    "targetMeanPrice": 327.6544,
    "targetLowPrice": 180.0,
}


def _fetch_recs(records):
    return lambda ticker: records


def _fetch_info(info):
    return lambda ticker: info


def _assert_not_called(*_args, **_kwargs):
    raise AssertionError("must not be called for an ineligible ticker")


# --- Eligibility -------------------------------------------------------

def test_etf_is_not_eligible_and_makes_no_fetch_call():
    result = get_analyst_view("VOO", fetch_recommendations=_assert_not_called, fetch_info=_assert_not_called)
    assert result is None


def test_unknown_ticker_fails_closed_and_makes_no_fetch_call():
    result = get_analyst_view("ZZZZ_NOPE", fetch_recommendations=_assert_not_called, fetch_info=_assert_not_called)
    assert result is None


def test_sgov_fixed_income_etf_is_not_eligible():
    result = get_analyst_view("SGOV", fetch_recommendations=_assert_not_called, fetch_info=_assert_not_called)
    assert result is None


# --- Consensus + target happy path --------------------------------------

def test_eligible_stock_with_full_data_returns_consensus_and_target():
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(_NVDA_INFO))
    assert view is not None
    assert view.consensus.buy == 57  # strongBuy(9) + buy(48)
    assert view.consensus.hold == 2
    assert view.consensus.sell == 1  # sell(1) + strongSell(0)
    assert view.consensus.total == 60
    assert view.consensus.label == "强力买入"
    assert view.target.current == 218.68
    assert view.target.high == 515.0
    assert view.target.mean == 327.6544
    assert view.target.low == 180.0


def test_upside_downside_percentages_calculated_correctly():
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(_NVDA_INFO))
    current = 218.68
    assert abs(view.target.high_pct - ((515.0 / current - 1) * 100)) < 1e-9
    assert abs(view.target.mean_pct - ((327.6544 / current - 1) * 100)) < 1e-9
    assert abs(view.target.low_pct - ((180.0 / current - 1) * 100)) < 1e-9


def test_negative_downside_renders_correctly():
    info = dict(_NVDA_INFO, currentPrice=600.0)  # low (180) now below current -> negative
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.target.low_pct < 0


# --- Native consensus label mapping -------------------------------------

def test_native_consensus_label_mapping_buy():
    info = dict(_NVDA_INFO, recommendationKey="buy")
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.consensus.label == "买入"


def test_unrecognized_recommendation_key_omits_label_but_keeps_counts():
    info = dict(_NVDA_INFO, recommendationKey="none")
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.consensus.label is None
    assert view.consensus.buy == 57


def test_missing_recommendation_key_omits_label_but_keeps_counts():
    info = {k: v for k, v in _NVDA_INFO.items() if k != "recommendationKey"}
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.consensus.label is None
    assert view.consensus.buy == 57


# --- Fail-closed: consensus ----------------------------------------------

def test_missing_analyst_counts_do_not_render_fake_zero_values():
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(None), fetch_info=_fetch_info(_NVDA_INFO))
    assert view.consensus is None
    assert view.target is not None  # independent failure


def test_missing_zero_period_row_fails_closed():
    records = [{"period": "-1m", "strongBuy": 9, "buy": 48, "hold": 2, "sell": 1, "strongSell": 0}]
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(records), fetch_info=_fetch_info(_NVDA_INFO))
    assert view.consensus is None


def test_all_zero_counts_fails_closed_not_rendered_as_zero():
    records = [{"period": "0m", "strongBuy": 0, "buy": 0, "hold": 0, "sell": 0, "strongSell": 0}]
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(records), fetch_info=_fetch_info(_NVDA_INFO))
    assert view.consensus is None


def test_negative_count_fails_closed():
    records = [{"period": "0m", "strongBuy": -1, "buy": 48, "hold": 2, "sell": 1, "strongSell": 0}]
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(records), fetch_info=_fetch_info(_NVDA_INFO))
    assert view.consensus is None


def test_recommendations_fetch_exception_fails_closed_for_consensus_only():
    def _raise(ticker):
        raise RuntimeError("network down")

    view = get_analyst_view("NVDA", fetch_recommendations=_raise, fetch_info=_fetch_info(_NVDA_INFO))
    assert view.consensus is None
    assert view.target is not None


# --- Fail-closed: target ---------------------------------------------------

def test_target_high_mean_low_render_correctly():
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(_NVDA_INFO))
    assert (view.target.low, view.target.mean, view.target.high) == (180.0, 327.6544, 515.0)


def test_malformed_target_ordering_fails_closed():
    info = dict(_NVDA_INFO, targetLowPrice=600.0)  # low > high, malformed
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.target is None
    assert view.consensus is not None  # independent failure


def test_missing_current_price_fails_closed_for_target():
    info = {k: v for k, v in _NVDA_INFO.items() if k not in ("currentPrice",)}
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.target is None


def test_zero_current_price_fails_closed_for_target():
    info = dict(_NVDA_INFO, currentPrice=0.0)
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.target is None


def test_missing_currency_fails_closed_for_target():
    info = {k: v for k, v in _NVDA_INFO.items() if k != "currency"}
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_fetch_info(info))
    assert view.target is None


def test_info_fetch_exception_fails_closed_for_target_only():
    def _raise(ticker):
        raise RuntimeError("network down")

    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(_NVDA_RECORDS), fetch_info=_raise)
    assert view.target is None
    assert view.consensus is not None


def test_both_subsections_failing_returns_none_view():
    view = get_analyst_view("NVDA", fetch_recommendations=_fetch_recs(None), fetch_info=_fetch_info(None))
    assert view is None


# --- Uses canonical resolved ticker identity -------------------------------

def test_uses_canonical_resolved_ticker_for_fetch_calls():
    seen = []

    def _rec(ticker):
        seen.append(("rec", ticker))
        return _NVDA_RECORDS

    def _info(ticker):
        seen.append(("info", ticker))
        return _NVDA_INFO

    get_analyst_view("NVDA", fetch_recommendations=_rec, fetch_info=_info)
    assert seen == [("rec", "NVDA"), ("info", "NVDA")]
