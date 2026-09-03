"""P0 investigation result: Page 2 showed "no verified ETF look-through
results" for the CBIL/ZSP/FINN/VDY/XSB/NVDA/VHT portfolio not because of
a ticker-alias mismatch (ETF_META/ETF_HOLDINGS already key by the same
bare/display ticker as Position.ticker -- there is no separate resolved-
market-ticker namespace collision there), but because ETF_HOLDINGS simply
never had Top-N constituent data for ZSP, FINN, VDY, or VHT at all.

ZSP tracks the identical S&P 500 index as the already-verified VOO/SPY/IVV
entries, so it was added with that same verified data (core/reference.py).
FINN (an actively-managed fund), VDY, and VHT have no verified holdings
snapshot available and correctly stay "uncovered" -- fail closed, not
guessed. These tests lock in that exact behavior.
"""

from __future__ import annotations

import pytest

from core.analytics import analyze, build_snapshot
from core.models import HoldingInput
from core.reference import ETF_HOLDINGS, ETF_META
from core.security_profile import resolve_security_profile
from core.ticker_resolution import CANADIAN_REFERENCE


def test_zsp_alias_is_the_same_canonical_identity_across_all_reference_data(quote_factory):
    """D: ZSP / ZSP.TO, VDY / VDY.TO, FINN / FINN.NE must all describe the
    ONE canonical security across ticker_resolution, ETF_META/ETF_HOLDINGS,
    and Security Profile -- never a namespace mismatch."""
    for ticker in ("ZSP", "VDY", "FINN", "XSB", "CBIL"):
        assert ticker in CANADIAN_REFERENCE, f"{ticker} missing from the resolver reference"
        assert ticker in ETF_META, f"{ticker} missing from ETF_META (same bare-ticker namespace)"
        assert resolve_security_profile(ticker) is not None, f"{ticker} missing a Security Profile"
    # The reference-data namespace never uses the resolved market symbol
    # (e.g. "ZSP.TO") as a key -- always the bare/display ticker.
    for suffixed in ("ZSP.TO", "VDY.TO", "FINN.NE", "XSB.TO", "CBIL.TO"):
        assert suffixed not in ETF_META
        assert suffixed not in ETF_HOLDINGS


def test_zsp_has_verified_holdings_and_contributes_to_nvda_true_exposure(quote_factory):
    """A: with ZSP's verified holdings, NVDA True Exposure = NVDA Direct + NVDA Via ZSP."""
    holdings = [HoldingInput("ZSP", 100), HoldingInput("NVDA", 10)]
    result = analyze(build_snapshot(holdings, quote_factory("ZSP", "NVDA")))
    nvda = next(x for x in result.true_exposures if x.ticker == "NVDA")
    assert nvda.indirect > 0
    assert nvda.true == pytest.approx(nvda.direct + nvda.indirect)
    assert "ZSP" not in result.uncovered_etfs


def test_partial_coverage_still_surfaces_the_verified_part(quote_factory):
    """B: VDY + VHT (unverified) + NVDA (direct, no ETF involved) -- the
    portfolio has no verified equity-ETF look-through at all here (VDY/VHT
    both uncovered), so true_exposures must reflect only NVDA's own direct
    weight -- never a guessed contribution from VDY/VHT."""
    holdings = [HoldingInput("VDY", 50), HoldingInput("VHT", 50), HoldingInput("NVDA", 10)]
    result = analyze(build_snapshot(holdings, quote_factory("VDY", "VHT", "NVDA")))
    assert {"VDY", "VHT"} <= set(result.uncovered_etfs)
    nvda = next(x for x in result.true_exposures if x.ticker == "NVDA")
    assert nvda.indirect == 0
    assert nvda.true == pytest.approx(nvda.direct)


def test_partial_coverage_with_one_verified_and_one_unverified_etf(quote_factory):
    """B (extended): mixing a verified ETF (ZSP) with an unverified one
    (VHT) must still show ZSP's verified contribution -- partial coverage,
    not zero coverage."""
    holdings = [HoldingInput("ZSP", 100), HoldingInput("VHT", 100), HoldingInput("NVDA", 10)]
    result = analyze(build_snapshot(holdings, quote_factory("ZSP", "VHT", "NVDA")))
    assert "VHT" in result.uncovered_etfs
    assert "ZSP" not in result.uncovered_etfs
    nvda = next(x for x in result.true_exposures if x.ticker == "NVDA")
    assert nvda.indirect > 0  # from ZSP alone


def test_fixed_income_etfs_never_produce_equity_lookthrough(quote_factory):
    """C: CBIL and XSB are Fixed Income, not equity look-through targets --
    they must never contribute to true_exposures or appear as uncovered
    equity ETFs."""
    holdings = [HoldingInput("CBIL", 100), HoldingInput("XSB", 100)]
    result = analyze(build_snapshot(holdings, quote_factory("CBIL", "XSB")))
    assert result.true_exposures == ()
    assert result.uncovered_etfs == ()


def test_unknown_etf_holdings_fail_closed_never_guessed(quote_factory):
    """E: FINN has no verified Top-N holdings snapshot -- it must be
    reported as uncovered, never assigned invented constituent weights."""
    holdings = [HoldingInput("FINN", 100)]
    result = analyze(build_snapshot(holdings, quote_factory("FINN")))
    assert result.uncovered_etfs == ("FINN",)
    assert result.true_exposures == ()
    assert "FINN" not in ETF_HOLDINGS


def test_exact_reported_portfolio_has_partial_verified_lookthrough(quote_factory):
    """The exact P0 portfolio: ZSP is now verified and contributes NVDA/AAPL/
    etc true exposure; FINN/VDY/VHT stay correctly uncovered (fail closed).
    Page 2 must therefore show partial results, never the old blanket
    "no verified ETF look-through" message."""
    holdings = [
        HoldingInput("CBIL", 3218), HoldingInput("ZSP", 1250), HoldingInput("FINN", 3500),
        HoldingInput("VDY", 900), HoldingInput("XSB", 2500), HoldingInput("NVDA", 150), HoldingInput("VHT", 100),
    ]
    result = analyze(build_snapshot(holdings, quote_factory("CBIL", "ZSP", "FINN", "VDY", "XSB", "NVDA", "VHT"), account_currency="CAD", usd_cad=1.36))
    exposure_with_indirect = [x for x in result.true_exposures if x.indirect > 0]
    assert exposure_with_indirect, "ZSP's verified holdings must produce at least one indirect exposure"
    assert {"FINN", "VDY", "VHT"} <= set(result.uncovered_etfs)
    assert "ZSP" not in result.uncovered_etfs
