"""Step 2B.1 -- Portfolio Intelligence Foundation.

Workstream A (ETF holdings coverage expansion), B (Risk Confidence /
scope-consistent risk semantics), and C (Fixed Income vs Cash-like asset
classification). Deterministic and network-free throughout -- no live
network call, no Gemini/Anthropic call anywhere in this module or the code
it tests.
"""

from __future__ import annotations

import json

import pytest

from core.analytics import analyze, build_snapshot, canonical_fact_packet
from core.models import HoldingInput, Quote
from core.reference import ETF_HOLDINGS


def _quotes(*tickers):
    return {t: Quote(t, 100.0, "USD", 99.0, "test", "verified") for t in tickers}


# ============================================================================
# Workstream A -- ETF holdings coverage expansion
# ============================================================================

def test_existing_covered_etf_data_unchanged():
    """VOO/XLK's already-verified reference holdings must be byte-for-byte
    unchanged by this step -- coverage expansion adds new ETFs, it never
    edits an already-verified one."""
    assert ETF_HOLDINGS["VOO"]["AAPL"] == pytest.approx(.070)
    assert ETF_HOLDINGS["VOO"]["MSFT"] == pytest.approx(.066)
    assert ETF_HOLDINGS["XLK"] == {"AAPL": .150, "MSFT": .130, "NVDA": .120, "AVGO": .065, "AMD": .020}


def test_zsp_not_duplicated_and_matches_voo():
    """ZSP was already covered before this step (same S&P 500 constituents
    as VOO/SPY/IVV) -- the audit must not add a second, divergent ZSP
    entry."""
    assert ETF_HOLDINGS["ZSP"] == ETF_HOLDINGS["VOO"]


def test_newly_covered_equity_etfs_use_verified_source_data():
    """VDY and VHT are new in this step, sourced from official issuer
    factsheets (Vanguard Investments Canada Inc. / The Vanguard Group,
    Inc.) -- their top-10 totals must match the exact published figures,
    not a rounded/estimated/inferred approximation."""
    assert sum(ETF_HOLDINGS["VDY"].values()) == pytest.approx(.693, abs=1e-9)
    assert sum(ETF_HOLDINGS["VHT"].values()) == pytest.approx(.523, abs=1e-9)
    # Exact top holding per the official factsheet.
    assert ETF_HOLDINGS["VDY"]["RY"] == pytest.approx(.159)
    assert ETF_HOLDINGS["VHT"]["LLY"] == pytest.approx(.142)


def test_uncovered_etf_remains_uncovered_when_data_cannot_be_verified():
    """FINN's official Fidelity factsheet discloses only an aggregate top-10
    weight (57.4%), never individual per-holding weights -- it must remain
    absent from ETF_HOLDINGS (fail closed) rather than an invented split."""
    assert "FINN" not in ETF_HOLDINGS


def test_fixed_income_etfs_never_counted_as_failed_equity_coverage(quote_factory):
    """A5: coverage is an EQUITY-ETF-only metric -- CBIL/XSB (Fixed Income /
    Cash-like) must never appear in uncovered_etfs even with zero look-
    through data, and must never degrade lookthrough_coverage."""
    holdings = [HoldingInput("CBIL", 50), HoldingInput("XSB", 50)]
    result = analyze(build_snapshot(holdings, quote_factory("CBIL", "XSB")))
    assert result.uncovered_etfs == ()
    assert result.lookthrough_coverage == "complete"
    assert result.risk_confidence == "High"


def test_page1_top_holdings_and_page2_lookthrough_share_verified_vdy_data():
    """A4/A8: Page 1 (core.etf_holdings) and Page 2 (core.analytics) must
    read the exact same newly-added VDY entry -- one source of truth."""
    from core.etf_holdings import get_etf_top_holdings

    top_holdings = get_etf_top_holdings("VDY")
    assert top_holdings is not None
    ry_row = next(h for h in top_holdings.holdings if h.ticker == "RY")
    assert ry_row.weight == ETF_HOLDINGS["VDY"]["RY"]


def test_true_exposure_lower_bound_semantics_preserved_for_new_coverage(quote_factory):
    """A9/A10: a newly-covered ETF's constituent contribution is still an
    identified, non-guessed figure -- VDY's own 15.9% RY weight combined
    with VDY's portfolio weight must equal RY's indirect exposure exactly,
    no proxy/estimated component mixed in."""
    holdings = [HoldingInput("VDY", 40), HoldingInput("NVDA", 10)]
    result = analyze(build_snapshot(holdings, quote_factory("VDY", "NVDA")))
    vdy_weight = next(p.weight for p in result.snapshot.complete_positions if p.ticker == "VDY")
    ry = next(x for x in result.true_exposures if x.ticker == "RY")
    assert ry.direct == 0
    assert ry.indirect == pytest.approx(vdy_weight * ETF_HOLDINGS["VDY"]["RY"])
    assert ry.true == ry.indirect


def test_no_new_provider_or_scraper_introduced():
    """Verify the coverage-expansion code path imports only the existing
    canonical modules -- no new HTTP client, scraper, or AI provider."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "reference.py").read_text(encoding="utf-8")
    for forbidden in ("requests", "httpx", "BeautifulSoup", "google", "genai", "anthropic", "playwright", "selenium"):
        assert forbidden not in source.lower()


# ============================================================================
# Workstream B -- Risk Confidence / scope-consistent risk semantics
# ============================================================================

def test_canonical_risk_confidence_field_exists(mixed_result):
    assert hasattr(mixed_result, "risk_confidence")
    assert mixed_result.risk_confidence in ("High", "Medium", "Limited")


def test_risk_confidence_is_deterministic(quote_factory):
    holdings = [HoldingInput("NVDA", 10), HoldingInput("VOO", 90)]
    a = analyze(build_snapshot(holdings, quote_factory("NVDA", "VOO")))
    b = analyze(build_snapshot(holdings, quote_factory("NVDA", "VOO")))
    assert a.risk_confidence == b.risk_confidence


def test_risk_confidence_maps_from_canonical_lookthrough_coverage(quote_factory):
    """B4: reuse the existing complete/partial/insufficient thresholds --
    no new threshold invented."""
    complete = analyze(build_snapshot([HoldingInput("VOO", 100)], quote_factory("VOO")))
    assert complete.lookthrough_coverage == "complete"
    assert complete.risk_confidence == "High"

    # XLV has no ETF_HOLDINGS entry; a small uncovered weight -> partial.
    partial_holdings = [HoldingInput("VOO", 90), HoldingInput("XLV", 10)]
    partial = analyze(build_snapshot(partial_holdings, quote_factory("VOO", "XLV")))
    assert partial.lookthrough_coverage == "partial"
    assert partial.risk_confidence == "Medium"

    # A large uncovered weight -> insufficient.
    insufficient_holdings = [HoldingInput("VOO", 20), HoldingInput("XLV", 80)]
    insufficient = analyze(build_snapshot(insufficient_holdings, quote_factory("VOO", "XLV")))
    assert insufficient.lookthrough_coverage == "insufficient"
    assert insufficient.risk_confidence == "Limited"


def test_risk_confidence_never_changes_risk_level(quote_factory):
    """B2/B9: confidence and risk level are independent dimensions -- a
    portfolio with materially incomplete coverage but no triggered risk
    flags must still report risk_level == "Measured", never auto-upgraded
    because confidence is limited."""
    holdings = [HoldingInput("JPM", 10), HoldingInput("XOM", 10), HoldingInput("XLV", 70)]
    result = analyze(build_snapshot(holdings, quote_factory("JPM", "XOM", "XLV"), cash=1000))
    assert result.risk_confidence == "Limited"
    assert result.risk_level == "Measured"
    assert not result.risk_flags


def test_fact_packet_carries_canonical_risk_confidence(mixed_result):
    """B3: Page 3's AI fact packet must read the same canonical field Page 1
    reads -- never a separately recomputed confidence."""
    packet = canonical_fact_packet(mixed_result)
    assert packet["portfolio"]["risk_confidence"] == mixed_result.risk_confidence
    assert packet["portfolio"]["risk_confidence"] in ("High", "Medium", "Limited")


def test_prompt_explains_risk_confidence_as_separate_dimension_from_risk_level():
    from core.ai import _prompt

    text = _prompt({})
    assert "risk_confidence" in text
    assert "较高" in text and "中等" in text and "受限" in text
    assert "separate" in text.lower() or "SEPARATE" in text


def test_prompt_forbids_raw_internal_token_leakage():
    from core.ai import _prompt

    text = _prompt({})
    assert "raw" in text.lower()
    assert "risk_confidence" in text  # named as the field never to print raw


def test_prompt_forbids_equating_limited_confidence_with_higher_or_lower_risk():
    from core.ai import _prompt

    text = _prompt({})
    normalized = " ".join(text.split())
    assert "never" in normalized.lower()
    assert "confidence describes evidence completeness" in normalized.lower() or "evidence completeness" in normalized.lower()


# ============================================================================
# Workstream C -- Asset Classification V2 (Fixed Income vs Cash-like)
# ============================================================================

def test_all_four_buckets_separate(quote_factory):
    holdings = [
        HoldingInput("NVDA", 30), HoldingInput("XSB", 20), HoldingInput("SGOV", 20),
        HoldingInput("GLD", 10),
    ]
    result = analyze(build_snapshot(holdings, quote_factory("NVDA", "XSB", "SGOV", "GLD"), cash=20))
    alloc = result.asset_allocation
    assert alloc["Equity"] > 0
    assert alloc["Fixed Income"] > 0
    assert alloc["Cash-like"] > 0
    assert alloc["Cash"] > 0
    assert alloc["Other"] > 0


def test_cbil_and_sgov_classified_cash_like(quote_factory):
    result = analyze(build_snapshot(
        [HoldingInput("CBIL", 5), HoldingInput("SGOV", 5), HoldingInput("NVDA", 90)],
        quote_factory("CBIL", "SGOV", "NVDA"),
    ))
    assert result.asset_allocation["Cash-like"] == pytest.approx(.10)
    assert "Fixed Income" not in result.asset_allocation


def test_xsb_classified_fixed_income_not_cash_like(quote_factory):
    result = analyze(build_snapshot(
        [HoldingInput("XSB", 10), HoldingInput("NVDA", 90)],
        quote_factory("XSB", "NVDA"),
    ))
    assert result.asset_allocation["Fixed Income"] == pytest.approx(.10)
    assert "Cash-like" not in result.asset_allocation


def test_asset_weights_reconcile_to_total_with_no_double_counting(quote_factory):
    holdings = [
        HoldingInput("NVDA", 30), HoldingInput("XSB", 20), HoldingInput("SGOV", 20), HoldingInput("GLD", 10),
    ]
    result = analyze(build_snapshot(holdings, quote_factory("NVDA", "XSB", "SGOV", "GLD"), cash=20))
    assert sum(result.asset_allocation.values()) == pytest.approx(1.0, abs=1e-9)


def test_fact_packet_asset_allocation_carries_cash_like_key(quote_factory):
    """C5: the generic asset_allocation dict already sent to the AI
    automatically carries the new "Cash-like" key -- no new top-level fact-
    packet field was needed."""
    result = analyze(build_snapshot(
        [HoldingInput("SGOV", 30), HoldingInput("NVDA", 70)], quote_factory("SGOV", "NVDA"),
    ))
    packet = canonical_fact_packet(result)
    assert packet["asset_allocation"]["Cash-like"] == pytest.approx(.30)


def test_prompt_forbids_duration_and_hedge_claims_for_cash_like_or_fixed_income():
    from core.ai import _prompt

    text = _prompt({})
    for forbidden_example in ("零久期", "提供股市对冲", "充分防御能力"):
        assert forbidden_example in text  # named explicitly as forbidden


def test_validate_semantic_overreach_fails_closed_on_zero_duration_claim():
    from core.ai import CommitteeResult, _validate_semantic_overreach_language
    from tests.test_ai import VALID

    payload = json.loads(json.dumps(VALID))
    payload["main_concern"] = "现金类资产零久期，因此完全不受利率波动影响。"
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="defensive-behavior"):
        _validate_semantic_overreach_language(committee)


def test_validate_semantic_overreach_fails_closed_on_defensive_hedge_claim():
    from core.ai import CommitteeResult, _validate_semantic_overreach_language
    from tests.test_ai import VALID

    payload = json.loads(json.dumps(VALID))
    payload["chairman_decision"] = "现金类与固定收益资产提供股市对冲，充分防御能力已经具备。"
    committee = CommitteeResult.model_validate(payload)
    with pytest.raises(ValueError, match="defensive-behavior"):
        _validate_semantic_overreach_language(committee)


def test_portfolio_insights_asset_structure_uses_factual_classification_only():
    from core.portfolio_insights import _asset_structure_insight

    insight = _asset_structure_insight({"Equity": 0.60, "Fixed Income": 0.15, "Cash-like": 0.20, "Cash": 0.05})
    text = insight["text"]
    for forbidden in ("零久期", "对冲", "均衡防御", "最优配置", "应维持"):
        assert forbidden not in text
