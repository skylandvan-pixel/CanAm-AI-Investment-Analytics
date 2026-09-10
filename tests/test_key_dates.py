"""Step 2A.5/2A.7 -- Security Key Dates.

Layer 1, deterministic and network-free by construction wherever possible:
FOMC/CPI/NFP are pure local table lookups (no mocking needed); earnings uses
a dependency-injected fetch_calendar so these tests never make a live
network call and never depend on the real world's earnings calendar
drifting over time.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.key_dates import (
    SecurityKeyDate,
    etf_is_us_market_exposed,
    format_event_date,
    get_security_key_dates,
    next_cpi_release_date,
    next_earnings_date,
    next_fomc_decision_date,
    next_nfp_release_date,
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
    """Step 2A.7: NVDA with an injected earnings date has 4 eligible
    candidates (NFP 9/4, CPI 9/11, FOMC 9/16, earnings 11/17) -- priority
    (earnings > FOMC > CPI > NFP) selects earnings+FOMC+CPI and drops NFP,
    then the selected set is rendered chronologically (CPI, FOMC, earnings)
    -- never in priority order."""
    events = get_security_key_dates(
        "NVDA", today=TODAY,
        fetch_calendar=lambda t: {"Earnings Date": [date(2026, 11, 17)]},
    )
    assert [e.event_type for e in events] == ["cpi", "fomc", "earnings"]
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
    """Step 2A.7: SPMO (a U.S. equity ETF) is macro-relevant and gets all
    three eligible macro releases (FOMC/CPI/NFP) -- but the earnings
    fetch_calendar callable must never even be invoked for it."""
    events = get_security_key_dates(
        "SPMO", today=TODAY,
        fetch_calendar=lambda t: pytest.fail("SPMO must never trigger an earnings fetch"),
    )
    assert len(events) == 3
    assert {e.event_type for e in events} == {"fomc", "cpi", "nfp"}
    assert "earnings" not in {e.event_type for e in events}


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


# ============================================================================
# Step 2A.7 -- Key Dates V2: CPI + NFP
#
# Adds exactly two new macro event families (U.S. CPI, U.S. Nonfarm
# Payrolls), both sourced from small local reference tables of officially
# published BLS release dates -- no network dependency, no heuristic
# ("first Friday", "second Tuesday") date generation. FOMC/earnings V1
# behavior and semantics are otherwise unchanged; the only structural change
# is that _finalize now selects which events survive a >3-candidate set by
# priority (earnings > FOMC > CPI > NFP) before sorting the survivors
# chronologically for display.
# ============================================================================

# --- CPI / NFP official-date lookups --------------------------------------------

def test_cpi_official_date_included_when_inside_90_day_horizon():
    assert next_cpi_release_date(TODAY) == date(2026, 9, 11)


def test_nfp_official_date_included_when_inside_90_day_horizon():
    assert next_nfp_release_date(TODAY) == date(2026, 10, 2)


def test_cpi_date_outside_90_day_horizon_excluded():
    far_future = date(2030, 1, 1)
    assert next_cpi_release_date(far_future) is None


def test_nfp_date_outside_90_day_horizon_excluded():
    far_future = date(2030, 1, 1)
    assert next_nfp_release_date(far_future) is None


def test_past_cpi_date_excluded():
    """A CPI date strictly before `today` must never be selected as "next"."""
    day_after_september_cpi = date(2026, 9, 12)  # the day after 2026-09-11
    assert next_cpi_release_date(day_after_september_cpi) != date(2026, 9, 11)
    assert next_cpi_release_date(day_after_september_cpi) == date(2026, 10, 14)


def test_past_nfp_date_excluded():
    day_after_september_nfp = date(2026, 9, 5)  # the day after 2026-09-04
    assert next_nfp_release_date(day_after_september_nfp) != date(2026, 9, 4)
    assert next_nfp_release_date(day_after_september_nfp) == date(2026, 10, 2)


def test_no_heuristic_generated_cpi_or_nfp_dates():
    """Every CPI/NFP date this module can ever return must come from the
    local official-date tables, never a "first Friday"/"second Tuesday"-
    style calendar rule. Probes a full year of "today" values and asserts
    every non-None result is a literal member of the reference tables."""
    from core.key_dates import _CPI_RELEASE_DATES, _NFP_RELEASE_DATES
    from datetime import timedelta

    probe_start = date(2025, 10, 1)
    for offset in range(0, 365, 7):
        probe_day = probe_start + timedelta(days=offset)
        cpi = next_cpi_release_date(probe_day)
        if cpi is not None:
            assert cpi in _CPI_RELEASE_DATES
        nfp = next_nfp_release_date(probe_day)
        if nfp is not None:
            assert nfp in _NFP_RELEASE_DATES


def test_cpi_and_nfp_events_marked_confirmed(quote_factory):
    events = get_security_key_dates("SGOV", today=TODAY)
    cpi = next(e for e in events if e.event_type == "cpi")
    nfp = next((e for e in events if e.event_type == "nfp"), None)
    assert cpi.status == "confirmed"
    assert "预计" not in format_event_date(cpi)
    if nfp is not None:
        assert nfp.status == "confirmed"


def test_no_duplicate_macro_events(quote_factory):
    events = get_security_key_dates("SGOV", today=TODAY)
    keys = [(e.event_type, e.event_date) for e in events]
    assert len(keys) == len(set(keys))


# --- Selection priority: earnings > FOMC > CPI > NFP ----------------------------

def test_selection_priority_earnings_beats_all_macro_events():
    """When 4 candidates are eligible (earnings + FOMC + CPI + NFP all
    within the horizon), earnings must survive the max-3 cut."""
    today = date(2026, 8, 20)  # NFP 9/4, CPI 9/11, FOMC 9/16, earnings 11/18 all in [today, today+90]
    events = get_security_key_dates(
        "NVDA", today=today, fetch_calendar=lambda t: {"Earnings Date": [date(2026, 11, 18)]},
    )
    assert len(events) == 3
    assert {e.event_type for e in events} == {"earnings", "fomc", "cpi"}
    assert "nfp" not in {e.event_type for e in events}


def test_selection_priority_fomc_beats_cpi_and_nfp():
    """A U.S. equity ETF with no earnings and all 3 macro events eligible
    keeps exactly FOMC + CPI + NFP (only 3 candidates, nothing to drop) --
    confirms FOMC is never itself dropped in favor of CPI/NFP."""
    today = date(2026, 8, 20)
    events = get_security_key_dates("SPMO", today=today)
    assert {e.event_type for e in events} == {"fomc", "cpi", "nfp"}


def test_final_rendering_is_chronological_not_priority_order():
    """The task's own worked example: selected events (earnings, FOMC, CPI)
    must render in chronological order (CPI 9/11 < FOMC 9/16 < earnings
    11/18), never in priority order (earnings, FOMC, CPI)."""
    today = date(2026, 8, 20)
    events = get_security_key_dates(
        "NVDA", today=today, fetch_calendar=lambda t: {"Earnings Date": [date(2026, 11, 18)]},
    )
    assert [e.event_type for e in events] == ["cpi", "fomc", "earnings"]
    assert [e.event_date for e in events] == sorted(e.event_date for e in events)


# --- Security relevance with the expanded macro pool ----------------------------

def test_individual_us_stock_receives_earnings_and_all_macro_events():
    today = date(2026, 8, 20)
    events = get_security_key_dates(
        "META", today=today, fetch_calendar=lambda t: {"Earnings Date": [date(2026, 10, 28)]},
    )
    types = {e.event_type for e in events}
    assert "earnings" in types
    assert types & {"fomc", "cpi", "nfp"}  # at least one macro event also present


def test_us_equity_etf_receives_macro_events_but_no_constituent_earnings():
    events = get_security_key_dates("VHT", today=TODAY, fetch_calendar=lambda t: pytest.fail("no earnings fetch for an ETF"))
    assert len(events) > 0
    assert all(e.event_type in {"fomc", "cpi", "nfp"} for e in events)


def test_sgov_receives_all_three_macro_event_types_when_in_horizon():
    today = date(2026, 8, 20)  # all 3 macro dates fall within [today, today+90]
    events = get_security_key_dates("SGOV", today=today)
    assert {e.event_type for e in events} == {"fomc", "cpi", "nfp"}


def test_canadian_etf_receives_no_us_constituent_earnings_or_macro_events():
    events = get_security_key_dates("VDY", today=TODAY, fetch_calendar=lambda t: pytest.fail("no earnings fetch for an ETF"))
    assert events == []


# --- Empty-safe / schema-safety regressions --------------------------------------

def test_key_dates_module_has_no_llm_or_gemini_dependency():
    """Layer 1 must stay LLM-free: no import of core.ai or any LLM SDK."""
    import ast
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "core" / "key_dates.py"
    tree = ast.parse(module_path.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(m == "core.ai" or m.startswith("core.ai.") for m in modules)
    assert not any("google" in m or "anthropic" in m for m in modules)


def test_committee_result_schema_unchanged_by_key_dates_v2():
    """Step 2A.7 is a Page 1 / Layer 1 feature only -- it must never touch
    the Gemini-facing CommitteeResult schema. Byte count rebaselined to
    4072 in Step 2A.10.1 (CommitteeMember.stance dropped "维持配置" -- a
    deliberate, approved, unrelated schema change; see
    tests/test_ai.py::test_schema_bytes_match_step_2a_3_approved_baseline)."""
    import json

    from core.ai import CommitteeResult

    schema = CommitteeResult.model_json_schema()
    assert len(json.dumps(schema, ensure_ascii=False).encode("utf-8")) == 4072
    assert len(schema.get("$defs", {})) == 4
