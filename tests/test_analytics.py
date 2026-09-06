from __future__ import annotations

import math

import pytest

from core.analytics import analyze, build_snapshot, canonical_fact_packet, treemap_rows
from core.models import HoldingInput, Quote


def test_layer1_without_api_keys(monkeypatch, mixed_result):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert mixed_result.portfolio_score > 0


def test_weight_normalization(quote_factory):
    result = analyze(build_snapshot([HoldingInput("AAPL", 1), HoldingInput("MSFT", 3)], quote_factory("AAPL", "MSFT"), cash=100))
    assert sum(w for _, w in result.top_direct) + result.snapshot.cash_weight == pytest.approx(1)


def test_treemap_uses_direct_weights_only(mixed_result):
    direct = dict(mixed_result.top_direct)
    rows = {row["ticker"]: row["weight"] for row in treemap_rows(mixed_result, minimum_weight=0)}
    assert rows["VOO"] == direct["VOO"]
    assert rows["NVDA"] != next(x.true for x in mixed_result.true_exposures if x.ticker == "NVDA")


def test_small_treemap_positions_aggregate(quote_factory):
    holdings = [HoldingInput("AAPL", 97), HoldingInput("MSFT", 1), HoldingInput("NVDA", 1), HoldingInput("JPM", 1)]
    result = analyze(build_snapshot(holdings, quote_factory("AAPL", "MSFT", "NVDA", "JPM")))
    rows = treemap_rows(result, minimum_weight=.025)
    assert rows[-1]["ticker"] == "Other"
    assert rows[-1]["weight"] == pytest.approx(.03)


def test_concentration_ranking(quote_factory):
    result = analyze(build_snapshot([HoldingInput("AAPL", 5), HoldingInput("MSFT", 3), HoldingInput("JPM", 2)], quote_factory("AAPL", "MSFT", "JPM")))
    assert [t for t, _ in result.top_direct] == ["AAPL", "MSFT", "JPM"]
    assert result.top3_concentration == pytest.approx(1)


def test_direct_plus_indirect_equals_true(mixed_result):
    for exposure in mixed_result.true_exposures:
        assert exposure.direct + exposure.indirect == pytest.approx(exposure.true)


def test_etf_overlap_aggregates_without_double_count(quote_factory):
    result = analyze(build_snapshot([HoldingInput("NVDA", 10), HoldingInput("VOO", 50), HoldingInput("XLK", 40)], quote_factory("NVDA", "VOO", "XLK")))
    nvda = next(x for x in result.true_exposures if x.ticker == "NVDA")
    assert nvda.direct == pytest.approx(.10)
    assert nvda.indirect == pytest.approx(.50 * .066 + .40 * .120)
    assert nvda.true == pytest.approx(.181)


def test_missing_etf_holdings_fail_closed(quote_factory):
    result = analyze(build_snapshot([HoldingInput("VXUS", 2), HoldingInput("AAPL", 8)], quote_factory("VXUS", "AAPL")))
    assert result.lookthrough_coverage == "insufficient"
    assert result.uncovered_etfs == ("VXUS",)


def test_sgov_is_fixed_income_and_not_uncovered(mixed_result):
    assert mixed_result.asset_allocation["Fixed Income"] == pytest.approx(.30)
    assert "SGOV" not in mixed_result.uncovered_etfs


# ---------------------------------------------------------------------------
# Step 2A.4 -- Report Semantic Consistency: these tests pin the exact
# deterministic formulas behind data_quality.{lookthrough_covered_weight,
# lookthrough_uncovered_weight, lookthrough_coverage, valuation_coverage} so
# that this patch's prompt-only semantic clarification (see core.ai._prompt)
# can never be mistaken for -- or silently accompanied by -- a Layer 1
# calculation change. No formula below is new; every assertion already holds
# on the pre-2A.4 codebase.
# ---------------------------------------------------------------------------

def test_data_quality_formulas_are_pinned(mixed_result):
    """mixed_result = NVDA(20)/VOO(50)/SGOV(30) @ $100, no cash. VOO is the
    only ETF held and it has an ETF_HOLDINGS entry, so covered_weight is
    exactly VOO's direct weight, uncovered_weight is exactly zero, and
    coverage is "complete" -- SGOV is Fixed Income, not an equity ETF, so it
    never enters either figure. All 3 positions priced -> valuation_coverage
    (a position-COUNT ratio) is exactly 1.0."""
    packet = canonical_fact_packet(mixed_result)
    dq = packet["data_quality"]
    assert dq["valuation_coverage"] == pytest.approx(1.0)
    assert dq["lookthrough_covered_weight"] == pytest.approx(.5)  # VOO's direct weight only
    assert dq["lookthrough_uncovered_weight"] == pytest.approx(0.0)
    assert dq["lookthrough_coverage"] == "complete"


def test_covered_and_uncovered_weight_are_disjoint_not_complements(quote_factory):
    """Audit Finding B: lookthrough_covered_weight and
    lookthrough_uncovered_weight are disjoint slices of the equity-ETF
    allocation only -- never complements of each other or of 100%. Here
    covered=20% (VOO, has ETF_HOLDINGS data) + uncovered=30% (XLF, an equity
    ETF with NO ETF_HOLDINGS entry) = 50%, leaving the other 50% of the
    portfolio (10% NVDA stock + 40% CBIL Fixed Income) outside both figures
    entirely -- "uncovered = 100% - covered" would wrongly claim 80%."""
    result = analyze(build_snapshot(
        [HoldingInput("NVDA", 10), HoldingInput("VOO", 20), HoldingInput("XLF", 30), HoldingInput("CBIL", 40)],
        quote_factory("NVDA", "VOO", "XLF", "CBIL"),
    ))
    packet = canonical_fact_packet(result)
    dq = packet["data_quality"]
    assert dq["lookthrough_covered_weight"] == pytest.approx(.20)
    assert dq["lookthrough_uncovered_weight"] == pytest.approx(.30)
    recognized_equity_etf_weight = dq["lookthrough_covered_weight"] + dq["lookthrough_uncovered_weight"]
    assert recognized_equity_etf_weight == pytest.approx(.50)
    assert recognized_equity_etf_weight != pytest.approx(1.0)  # not complements of 100%
    assert (1.0 - dq["lookthrough_covered_weight"]) != pytest.approx(dq["lookthrough_uncovered_weight"])


def test_valuation_coverage_is_a_position_count_ratio_not_a_valuation_metric(quote_factory):
    """Audit Finding E: valuation_coverage is literally (# positions with a
    usable market value) / (# entered positions) -- a pricing/position-count
    ratio, unrelated in formula to any position's dollar size or to any
    fundamental valuation metric (P/E, P/B, yield, ...), none of which exist
    in this packet."""
    quotes = quote_factory("AAPL")
    quotes["MISSING"] = Quote("MISSING", None, None, status="error:test")
    snapshot = build_snapshot([HoldingInput("AAPL", 1), HoldingInput("MISSING", 50)], quotes)
    result = analyze(snapshot)
    packet = canonical_fact_packet(result)
    assert snapshot.coverage_ratio == pytest.approx(.5)  # 1 of 2 positions priced
    assert packet["data_quality"]["valuation_coverage"] == pytest.approx(.5)


def test_effective_n_semantics_are_distinct(mixed_result):
    assert mixed_result.direct_effective_n is not None
    assert mixed_result.lookthrough_effective_n is not None
    assert mixed_result.direct_effective_n != mixed_result.lookthrough_effective_n


def test_high_true_exposure_creates_risk_flag(quote_factory):
    result = analyze(build_snapshot([HoldingInput("NVDA", 8), HoldingInput("VOO", 2)], quote_factory("NVDA", "VOO")))
    assert result.risk_flags[0].key == "single_name"
    assert result.risk_flags[0].severity == "high"


def test_risk_flags_are_capped_at_three(mixed_result):
    assert len(mixed_result.risk_flags) <= 3


def test_canonical_content_is_deduplicated(mixed_result):
    locations = list(mixed_result.canonical_content.values())
    assert len(locations) == len(set(locations))


def test_fact_packet_is_aggregated_not_raw_holdings(mixed_result):
    packet = canonical_fact_packet(mixed_result)
    assert "quantity" not in str(packet)
    assert packet["portfolio"]["portfolio_score"] == mixed_result.portfolio_score


def test_partial_price_data_does_not_fabricate_value(quote_factory):
    quotes = quote_factory("AAPL")
    quotes["MISSING"] = Quote("MISSING", None, None, status="error:test")
    snapshot = build_snapshot([HoldingInput("AAPL", 1), HoldingInput("MISSING", 50)], quotes)
    assert snapshot.positions[1].market_value is None
    assert snapshot.total_assets == pytest.approx(100)


def test_invalid_currency_is_rejected(quote_factory):
    with pytest.raises(ValueError):
        build_snapshot([HoldingInput("AAPL", 1, currency="EUR")], quote_factory("AAPL"))


def test_account_type_is_a_single_portfolio_level_fact(quote_factory):
    snapshot = build_snapshot([HoldingInput("AAPL", 5)], quote_factory("AAPL"), account_type="TFSA")
    assert snapshot.account_type == "TFSA"


def test_account_type_defaults_to_taxable_when_unset(quote_factory):
    snapshot = build_snapshot([HoldingInput("AAPL", 5)], quote_factory("AAPL"))
    assert snapshot.account_type == "Taxable"


def test_account_type_never_blocks_analysis_when_unrecognized(quote_factory):
    snapshot = build_snapshot([HoldingInput("AAPL", 5)], quote_factory("AAPL"), account_type="RRSP")
    assert snapshot.account_type == "Taxable"


def test_account_type_included_in_fact_packet(mixed_result):
    packet = canonical_fact_packet(mixed_result)
    assert packet["portfolio"]["account_type"] in {"TFSA", "Taxable"}


def test_currency_falls_back_to_quote_currency_when_not_specified(quote_factory):
    """Per-row currency is gone from the UI; a holding with no explicit
    currency must still value correctly using the quote's real currency."""
    result = analyze(build_snapshot([HoldingInput("AAPL", 5)], quote_factory("AAPL"), account_currency="USD"))
    assert result.snapshot.positions[0].market_value == pytest.approx(500.0)
