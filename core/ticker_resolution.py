"""Deterministic Canada/U.S. ticker identity resolution.

A bare ticker symbol can collide across markets -- e.g. bare "FINN" is
an unrelated, thinly-traded U.S. OTC stock (First National of Nebraska,
Inc.), while the Canadian ETF the user actually means (Fidelity Global
Innovators ETF Series L) only exists on Yahoo as "FINN.NE". Querying the
bare symbol silently returns real-looking data for the wrong security.

This module is the single source of truth for "what market symbol does
this user-entered ticker actually mean" for both market-data fetching
(core/market.py) and the Security Profile feature (core/security_profile.py)
-- one ticker, one canonical identity, never two independently-maintained
mappings.
"""

from __future__ import annotations

from dataclasses import dataclass

CANADIAN_EXCHANGE_SUFFIXES = (".TO", ".V", ".NE")


class TickerIdentityError(Exception):
    """Raised when a ticker cannot be resolved to one verified security
    identity, or when the fetched quote's own metadata doesn't match the
    identity it was resolved to. Callers must fail closed: never guess,
    never substitute, never proceed with a mismatched security."""


@dataclass(frozen=True)
class SecurityIdentity:
    input_ticker: str
    resolved_ticker: str
    expected_currency: str | None = None
    expected_exchange: str | None = None
    expected_quote_type: str | None = None


# The verified local reference for bare Canadian symbols relevant to this
# MVP. A ticker listed here is ALWAYS queried under its mapped market
# symbol -- never the bare form -- so it can never collide with an
# unrelated security sharing the same bare ticker on another exchange.
# Suffixes are deliberately per-ticker (.TO/.V/.NE), never assumed.
CANADIAN_REFERENCE: dict[str, SecurityIdentity] = {
    "CBIL": SecurityIdentity("CBIL", "CBIL.TO", "CAD", "TOR", "ETF"),
    "ZSP": SecurityIdentity("ZSP", "ZSP.TO", "CAD", "TOR", "ETF"),
    "XSB": SecurityIdentity("XSB", "XSB.TO", "CAD", "TOR", "ETF"),
    "VDY": SecurityIdentity("VDY", "VDY.TO", "CAD", "TOR", "ETF"),
    "FINN": SecurityIdentity("FINN", "FINN.NE", "CAD", "NEO", "ETF"),
}


def resolve_ticker(raw_ticker: str) -> SecurityIdentity:
    """Resolve a user-entered ticker to its verified market identity.

    Priority: (1) an explicit exchange suffix the user already typed is
    respected exactly as-is; (2) the verified Canadian reference for a
    bare symbol; (3) otherwise passed through unchanged (a standard
    equity, typically U.S.) -- this never guesses a mapping that isn't
    verified, and downstream quote-metadata validation is the backstop
    for anything not covered by the reference.
    """
    ticker = raw_ticker.strip().upper()
    if any(ticker.endswith(suffix) for suffix in CANADIAN_EXCHANGE_SUFFIXES):
        return SecurityIdentity(ticker, ticker, "CAD")
    if ticker in CANADIAN_REFERENCE:
        return CANADIAN_REFERENCE[ticker]
    return SecurityIdentity(ticker, ticker)


def validate_identity(identity: SecurityIdentity, *, currency: str | None, quote_type: str | None) -> None:
    """Fail closed if the fetched quote's own metadata contradicts the
    identity we resolved to. A quote coming back is not, by itself,
    proof it's the right security -- an unrelated instrument on another
    exchange can share the same bare ticker and still return valid-
    looking data."""
    if identity.expected_currency and currency != identity.expected_currency:
        raise TickerIdentityError(f"{identity.input_ticker} 无法可靠识别证券身份，请确认代码或交易市场。")
    if identity.expected_quote_type and quote_type and quote_type != identity.expected_quote_type:
        raise TickerIdentityError(f"{identity.input_ticker} 无法可靠识别证券身份，请确认代码或交易市场。")
