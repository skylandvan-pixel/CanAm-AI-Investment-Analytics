"""Step 2A.11 -- ETF Top Holdings (Page 1 Security Profile).

Deterministic and network-free by construction: core.etf_holdings is a pure
lookup against core.reference.ETF_HOLDINGS, the exact same dataset
core.analytics already reads for Page 2 ETF Look-through. No mocking needed,
no live network call ever possible from this module.
"""

from __future__ import annotations

from core.etf_holdings import get_etf_top_holdings
from core.reference import ETF_HOLDINGS


# --- Eligibility ------------------------------------------------------------

def test_etf_with_verified_reference_holdings_renders():
    view = get_etf_top_holdings("VOO")
    assert view is not None
    assert view.ticker == "VOO"
    assert len(view.holdings) > 0


def test_individual_stock_does_not_render():
    assert get_etf_top_holdings("NVDA") is None
    assert get_etf_top_holdings("AAPL") is None


def test_etf_without_verified_reference_holdings_fails_closed():
    """VHT and SGOV are canonical ETFs (see core.security_profile) but have
    no entry in core.reference.ETF_HOLDINGS -- must fail closed, never a
    fabricated constituent list."""
    assert "VHT" not in ETF_HOLDINGS  # sanity: confirms the fixture premise
    assert get_etf_top_holdings("VHT") is None
    assert get_etf_top_holdings("SGOV") is None


def test_unknown_ticker_fails_closed():
    assert get_etf_top_holdings("ZZZZ_NOPE") is None


def test_no_fake_zero_holdings():
    """A None result must never be papered over with an empty-but-present
    ETFTopHoldings -- the caller (_etf_top_holdings_html) relies on None
    itself as the fail-closed signal."""
    assert get_etf_top_holdings("VHT") is None


# --- Cardinality / sorting ---------------------------------------------------

def test_maximum_ten_holdings_displayed():
    view = get_etf_top_holdings("VOO")  # VOO has 15 reference constituents
    assert view.total_reference_count == 15
    assert len(view.holdings) == 10


def test_holdings_sorted_by_weight_descending():
    view = get_etf_top_holdings("VOO")
    weights = [h.weight for h in view.holdings]
    assert weights == sorted(weights, reverse=True)


def test_deterministic_tie_ordering():
    """VOO's MSFT and NVDA are both weighted 0.066 -- the stable secondary
    ordering (ticker ascending) must always put MSFT before NVDA."""
    assert ETF_HOLDINGS["VOO"]["MSFT"] == ETF_HOLDINGS["VOO"]["NVDA"] == 0.066
    view = get_etf_top_holdings("VOO")
    tickers = [h.ticker for h in view.holdings]
    assert tickers.index("MSFT") < tickers.index("NVDA")


def test_tie_ordering_is_stable_across_repeated_calls():
    first = [h.ticker for h in get_etf_top_holdings("VOO").holdings]
    second = [h.ticker for h in get_etf_top_holdings("VOO").holdings]
    assert first == second


# --- Constituent weight vs portfolio exposure -------------------------------

def test_constituent_weights_are_etf_weights_not_portfolio_exposure():
    """The critical semantic guard: a constituent's displayed weight is its
    share of the ETF itself, exactly ETF_HOLDINGS[ticker][constituent] --
    never multiplied by any portfolio-level holding weight (this module
    never even receives a portfolio)."""
    view = get_etf_top_holdings("VOO")
    for row in view.holdings:
        assert row.weight == ETF_HOLDINGS["VOO"][row.ticker]


# --- Total-weight rule --------------------------------------------------------

def test_displayed_total_calculated_correctly():
    view = get_etf_top_holdings("VOO")
    expected = sum(h.weight for h in view.holdings)
    assert abs(view.displayed_total - expected) < 1e-12


def test_fewer_than_ten_holdings_reports_total_reference_count_below_ten():
    """XLK has only 5 reference constituents -- the UI layer uses
    total_reference_count < 10 to avoid the "前十大持仓合计" (Top-10) label;
    verified here at the data layer that the count is honestly reported."""
    view = get_etf_top_holdings("XLK")
    assert view.total_reference_count == 5
    assert len(view.holdings) == 5  # all of them shown, nothing truncated


def test_ten_or_more_reference_holdings_reports_full_count():
    view = get_etf_top_holdings("VOO")
    assert view.total_reference_count == 15
    assert view.total_reference_count >= 10


# --- Company name enrichment (reuses core.security_profile, no new source) --

def test_company_name_shown_when_reliably_available():
    view = get_etf_top_holdings("VOO")
    aapl = next(h for h in view.holdings if h.ticker == "AAPL")
    assert aapl.name == "Apple Inc."


def test_ticker_only_when_name_not_reliably_available():
    """Every current VOO/SPY/IVV/ZSP constituent happens to be in
    core.security_profile's stock snapshot, so this exercises the fallback
    path directly against the underlying accessor instead."""
    from core.security_profile import stock_name_en

    assert stock_name_en("ZZZZ_NOPE") is None


# --- Page 1 / Page 2 single source of truth ----------------------------------

def test_reconciles_with_page_2_lookthrough_source_data():
    """Page 1 Top Holdings and Page 2 True Exposure must derive from the
    exact same ETF_HOLDINGS entry -- Page 2's indirect contribution for a
    constituent is (ETF's own portfolio weight) x (constituent's ETF
    weight), so dividing back out must reproduce exactly what Page 1
    displays, proving there is one source of truth, not two pipelines."""
    from core.analytics import analyze, build_snapshot
    from core.models import HoldingInput, Quote

    holdings = [HoldingInput("VOO", 30), HoldingInput("SGOV", 70)]
    quotes = {
        "VOO": Quote("VOO", 100.0, "USD", 99.0, "test", "verified"),
        "SGOV": Quote("SGOV", 100.0, "USD", 99.0, "test", "verified"),
    }
    result = analyze(build_snapshot(holdings, quotes))
    voo_portfolio_weight = next(w for t, w in [(p.ticker, p.weight or 0) for p in result.snapshot.complete_positions] if t == "VOO")

    view = get_etf_top_holdings("VOO")
    aapl_row = next(h for h in view.holdings if h.ticker == "AAPL")
    aapl_exposure = next(x for x in result.true_exposures if x.ticker == "AAPL")

    assert abs(aapl_exposure.indirect - voo_portfolio_weight * aapl_row.weight) < 1e-9
    # And the Page 1 constituent weight itself is untouched by the portfolio
    # weight -- never displayed as the already-diluted indirect figure.
    assert aapl_row.weight != aapl_exposure.indirect


def test_uses_same_etf_holdings_dict_object_as_analytics():
    """core.etf_holdings must import the SAME ETF_HOLDINGS object
    core.analytics imports -- never a copy or a second dataset."""
    import core.analytics as analytics_module
    import core.etf_holdings as etf_holdings_module

    assert etf_holdings_module.ETF_HOLDINGS is analytics_module.ETF_HOLDINGS
