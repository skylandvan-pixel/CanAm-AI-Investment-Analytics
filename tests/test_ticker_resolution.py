from __future__ import annotations

import pytest

from core.ticker_resolution import CANADIAN_REFERENCE, TickerIdentityError, resolve_ticker, validate_identity


def test_bare_canadian_ticker_resolves_to_reference_symbol():
    identity = resolve_ticker("finn")
    assert identity.resolved_ticker == "FINN.NE"
    assert identity.expected_currency == "CAD"
    assert identity.expected_quote_type == "ETF"


def test_bare_canadian_ticker_never_assumes_dot_to_suffix():
    """CBIL/ZSP/XSB are .TO but FINN is .NE -- the resolver must use the
    per-ticker reference, not a blanket .TO assumption."""
    assert resolve_ticker("cbil").resolved_ticker == "CBIL.TO"
    assert resolve_ticker("finn").resolved_ticker == "FINN.NE"


def test_explicit_to_suffix_left_unchanged():
    identity = resolve_ticker("zsp.to")
    assert identity.resolved_ticker == "ZSP.TO"
    assert identity.input_ticker == "ZSP.TO"


def test_explicit_ne_suffix_left_unchanged():
    identity = resolve_ticker("finn.ne")
    assert identity.resolved_ticker == "FINN.NE"


def test_explicit_v_suffix_left_unchanged():
    identity = resolve_ticker("xyz.v")
    assert identity.resolved_ticker == "XYZ.V"


def test_us_ticker_left_unchanged_with_no_forced_expectation():
    identity = resolve_ticker("NVDA")
    assert identity.resolved_ticker == "NVDA"
    assert identity.expected_currency is None


def test_unmapped_bare_ticker_passes_through_without_guessing():
    identity = resolve_ticker("SOMEUNKNOWNTICKER")
    assert identity.resolved_ticker == "SOMEUNKNOWNTICKER"
    assert identity.expected_currency is None


def test_matching_identity_passes_validation():
    identity = resolve_ticker("FINN")
    validate_identity(identity, currency="CAD", quote_type="ETF")  # must not raise


def test_wrong_currency_candidate_is_rejected():
    """This is the exact collision that caused the P0 bug: bare "FINN"
    resolves (correctly) to FINN.NE, but if a caller ever fed back the
    unrelated bare-FINN quote's currency (USD), validation must reject it."""
    identity = resolve_ticker("FINN")
    with pytest.raises(TickerIdentityError, match="FINN"):
        validate_identity(identity, currency="USD", quote_type="EQUITY")


def test_wrong_quote_type_candidate_is_rejected():
    identity = resolve_ticker("ZSP")
    with pytest.raises(TickerIdentityError):
        validate_identity(identity, currency="CAD", quote_type="EQUITY")


def test_ambiguous_identity_fails_closed_not_guessed():
    """A resolved identity whose fetched metadata matches neither the
    expected currency nor quote type is ambiguous data, not a security
    to guess at -- it must always raise, never pick one silently."""
    identity = resolve_ticker("VDY")
    with pytest.raises(TickerIdentityError):
        validate_identity(identity, currency="USD", quote_type="EQUITY")


def test_unknown_currency_is_not_validated_against_a_forced_expectation():
    """A ticker outside the Canadian reference has no forced expectation,
    so ordinary U.S. tickers are unaffected by this guard."""
    identity = resolve_ticker("AAPL")
    validate_identity(identity, currency="USD", quote_type="EQUITY")  # must not raise


def test_canadian_reference_covers_the_mvp_tickers():
    assert {"CBIL", "ZSP", "XSB", "VDY", "FINN"} <= set(CANADIAN_REFERENCE)
    for identity in CANADIAN_REFERENCE.values():
        assert identity.expected_currency == "CAD"
        assert identity.resolved_ticker.endswith((".TO", ".V", ".NE"))
