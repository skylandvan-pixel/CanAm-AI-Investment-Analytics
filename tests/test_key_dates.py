"""Step 2A.5 -- Security Key Dates.

Layer 1, deterministic and network-free by construction wherever possible:
FOMC is a pure local table lookup (no mocking needed); earnings uses a
dependency-injected fetch_calendar so these tests never make a live network
call and never depend on the real world's earnings calendar drifting over
time.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.key_dates import (
    SecurityKeyDate,
    etf_is_us_market_exposed,
    format_event_date,
    get_security_key_dates,
    next_earnings_date,
    next_fomc_decision_date,
)

TODAY = date(2026, 9, 6)


# --- Sorting / deduplication pipeline (_finalize, via get_security_key_dates) --

def test_no_events_beyond_90_days():
    events = get_security_key_dates(
        "NVDA", today=TODAY,
        fetch_calendar=lambda t: {"Earnings Date": [date(2026, 12, 31)]},  # 116 days out
    )
    assert all(e.event_type != "earnings" for e in events)


def test_nearest_event_first():
    """NVDA's next FOMC (2026-09-16) is nearer than its injected earnings
    date (2026-11-17) -- the FOMC entry must come first."""
    events = get_security_key_dates(
        "NVDA", today=TODAY,
        fetch_calendar=lambda t: {"Earnings Date": [date(2026, 11, 17)]},
    )
    assert [e.event_type for e in events] == ["fomc", "earnings"]
    dates = [e.event_date for e in events]
    assert dates == sorted(dates)


def test_max_three_events():
    """Even a pathological event list is capped at 3 by _finalize."""
    from core.key_dates import _finalize

    events = [
        SecurityKeyDate(date(2026, 9, 10), "a", "t", "confirmed", "n", "s"),
        SecurityKeyDate(date(2026, 9, 12), "b", "t", "confirmed", "n", "s"),
        SecurityKeyDate(date(2026, 9, 14), "c", "t", "confirmed", "n", "s"),
        SecurityKeyDate(date(2026, 9, 16), "d", "t", "confirmed", "n", "s"),
    ]
    assert len(_finalize(events, TODAY)) == 3


def test_past_events_excluded():
    from core.key_dates import _finalize

    events = [SecurityKeyDate(date(2026, 9, 1), "earnings", "t", "confirmed", "n", "s")]  # before TODAY
    assert _finalize(events, TODAY) == []


def test_duplicates_excluded():
    from core.key_dates import _finalize

    events = [
        SecurityKeyDate(date(2026, 9, 16), "fomc", "t", "confirmed", "n", "s"),
        SecurityKeyDate(date(2026, 9, 16), "fomc", "t", "confirmed", "n", "s"),
    ]
    assert len(_finalize(events, TODAY)) == 1


# --- Individual stock: earnings -----------------------------------------------

def test_stock_next_earnings_included_when_valid():
    events = get_security_key_dates(
        "NVDA", today=TODAY,
        fetch_calendar=lambda t: {"Earnings Date": [date(2026, 11, 17)]},
    )
    earnings = [e for e in events if e.event_type == "earnings"]
    assert len(earnings) == 1
    assert earnings[0].event_date == date(2026, 11, 17)
    assert earnings[0].status == "estimated"


def test_etf_never_gets_a_company_earnings_event():
    events = get_security_key_dates(
        "SPMO", today=TODAY,
        # Even if a caller mistakenly supplied earnings-shaped data, the
        # orchestrator must never call fetch_calendar for a non-stock at all.
        fetch_calendar=lambda t: {"Earnings Date": [date(2026, 10, 1)]},
    )
    assert all(e.event_type != "earnings" for e in events)


# --- FOMC ----------------------------------------------------------------------

def test_fomc_correct_next_future_decision_selected():
    assert next_fomc_decision_date(TODAY) == date(2026, 9, 16)
    assert next_fomc_decision_date(date(2026, 9, 17)) == date(2026, 10, 28)


def test_fomc_none_beyond_reference_table_horizon():
    assert next_fomc_decision_date(date(2030, 1, 1)) is None


# --- SGOV / fixed-income --------------------------------------------------------

def test_sgov_shows_fomc_not_earnings():
    events = get_security_key_dates("SGOV", today=TODAY)
    assert any(e.event_type == "fomc" for e in events)
    assert all(e.event_type != "earnings" for e in events)


def test_canadian_fixed_income_etf_excluded_from_fomc():
    """CBIL/XSB are Canadian Treasury/bond ETFs sharing the "Fixed Income"
    bucket with genuinely U.S. SGOV -- must not get FOMC."""
    assert get_security_key_dates("CBIL", today=TODAY) == []
    assert get_security_key_dates("XSB", today=TODAY) == []


# --- SPMO ------------------------------------------------------------------------

def test_spmo_shows_fomc_no_arbitrary_constituent_earnings():
    events = get_security_key_dates(
        "SPMO", today=TODAY,
        fetch_calendar=lambda t: pytest.fail("SPMO must never trigger an earnings fetch"),
    )
    assert len(events) == 1
    assert events[0].event_type == "fomc"


# --- Provider failure --------------------------------------------------------

def test_provider_failure_omits_earnings_event():
    def _raise(_ticker):
        raise RuntimeError("simulated network failure")

    events = get_security_key_dates("NVDA", today=TODAY, fetch_calendar=_raise)
    assert all(e.event_type != "earnings" for e in events)
    assert any(e.event_type == "fomc" for e in events)  # profile stays functional


def test_malformed_calendar_shape_omits_earnings_event():
    events = get_security_key_dates("NVDA", today=TODAY, fetch_calendar=lambda t: {"Earnings Date": "not-a-list"})
    assert all(e.event_type != "earnings" for e in events)


def test_empty_calendar_omits_earnings_event():
    events = get_security_key_dates("NVDA", today=TODAY, fetch_calendar=lambda t: {})
    assert all(e.event_type != "earnings" for e in events)


# --- Canonical ticker resolution ----------------------------------------------

def test_earnings_fetch_uses_canonical_resolved_ticker():
    """FINN is a verified Canadian ETF (FINN.NE) in core.ticker_resolution,
    resolved away from the unrelated bare-FINN U.S. OTC stock -- but FINN's
    Security Profile classifies it as an ETF, so earnings must never even be
    attempted for it regardless of resolution."""
    seen = {}

    def _fetch(resolved_ticker):
        seen["ticker"] = resolved_ticker
        return {}

    get_security_key_dates("FINN", today=TODAY, fetch_calendar=_fetch)
    assert "ticker" not in seen  # ETF -- fetch_calendar never called


def test_next_earnings_date_resolves_canonical_identity_directly():
    seen = {}

    def _fetch(resolved_ticker):
        seen["ticker"] = resolved_ticker
        return {"Earnings Date": [date(2026, 10, 1)]}

    next_earnings_date("FINN", today=TODAY, fetch_calendar=_fetch)
    assert seen["ticker"] == "FINN.NE"  # canonical resolved identity, not the bare input


# --- Estimated vs confirmed date rendering -------------------------------------

def test_estimated_earnings_rendered_with_预计_prefix():
    events = get_security_key_dates(
        "NVDA", today=TODAY,
        fetch_calendar=lambda t: {"Earnings Date": [date(2026, 11, 17)]},
    )
    earnings = next(e for e in events if e.event_type == "earnings")
    assert earnings.status == "estimated"
    assert format_event_date(earnings).startswith("预计 ")


def test_confirmed_date_has_no_预计_prefix():
    confirmed = SecurityKeyDate(date(2026, 9, 16), "fomc", "美联储 FOMC 利率决议", "confirmed", "n", "fomc_reference")
    text = format_event_date(confirmed)
    assert text == "2026-09-16"
    assert "预计" not in text


def test_window_status_uses_month_third_not_false_precision():
    window = SecurityKeyDate(date(2026, 10, 21), "earnings", "t", "window", "n", "yfinance", window_end=date(2026, 10, 25))
    text = format_event_date(window)
    assert text == "预计 10月下旬"
    assert "2026-10-21" not in text  # never a fabricated single exact day


def test_fomc_decision_dates_are_confirmed_status():
    events = get_security_key_dates("SGOV", today=TODAY)
    fomc = next(e for e in events if e.event_type == "fomc")
    assert fomc.status == "confirmed"


# --- ETF market-exposure classification (regression for the ex-US substring bug) --

def test_us_equity_etf_is_market_exposed():
    assert etf_is_us_market_exposed("美国股票", "标普 500 大型股指数 ETF")


def test_zsp_cad_priced_us_index_is_market_exposed():
    assert etf_is_us_market_exposed("美国股票（加元计价）", "标普 500 大型股指数 ETF")


def test_us_treasury_fixed_income_is_market_exposed():
    assert etf_is_us_market_exposed("固定收益", "美国超短期国债")


def test_canadian_fixed_income_is_not_market_exposed():
    assert not etf_is_us_market_exposed("固定收益", "加拿大超短期国债")


def test_ex_us_international_fund_is_not_falsely_matched():
    """Regression: "全球（除美国）股票指数 ETF" (ex-US) literally contains the
    characters "美国" as part of a NEGATED phrase -- a naive substring check
    on the concatenated label text would wrongly mark this U.S.-exposed."""
    assert not etf_is_us_market_exposed("国际股票", "全球（除美国）股票指数 ETF")
    assert not etf_is_us_market_exposed("国际股票", "发达市场（除美加）股票指数 ETF")


def test_canadian_equity_etf_is_not_market_exposed():
    assert not etf_is_us_market_exposed("加拿大股票", "高股息 ETF")


def test_alternative_asset_etf_is_not_market_exposed():
    assert not etf_is_us_market_exposed("另类资产", "黄金 ETF")


# --- No profile / unknown ticker ------------------------------------------------

def test_unknown_ticker_returns_no_events():
    assert get_security_key_dates("ZZZZ", today=TODAY) == []
