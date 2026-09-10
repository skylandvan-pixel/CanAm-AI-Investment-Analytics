"""Step 2A.11 -- ETF Top Holdings (Page 1 Security Profile).

Layer 1, deterministic "what does this ETF mainly hold" reference for the
Security Profile card, reusing the EXACT SAME canonical reference-holdings
dataset (core.reference.ETF_HOLDINGS) that core.analytics already uses for
the Page 2 ETF Look-through / True Exposure engine -- one source of truth,
never a second holdings pipeline, no network call, no new data source.

Product semantics: this answers "what does the selected ETF itself hold"
(an ETF-level constituent weight), which is a different question from Page
2's True Exposure ("how much of my portfolio is indirectly exposed to this
name"). A constituent's weight here is always its share of the ETF itself
(e.g. LLY is ~1.1% of VOO), never multiplied by the ETF's own portfolio
weight -- that multiplication is exactly what core.analytics.analyze does
internally to produce Page 2's indirect/true exposure figures, and this
module never repeats or reimplements that calculation.

ETF_HOLDINGS is a frozen, non-exhaustive Top-N reference snapshot (see its
own docstring) -- an ETF absent from it, or any non-ETF ticker, fails closed
to None here, never a guessed/manufactured constituent list.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.reference import ETF_HOLDINGS
from core.security_profile import resolve_security_profile, stock_name_en

_MAX_HOLDINGS = 10


@dataclass(frozen=True)
class ETFHoldingRow:
    ticker: str
    name: str | None  # company/security name if reliably available, else None
    weight: float  # this constituent's weight WITHIN the ETF, never portfolio-level


@dataclass(frozen=True)
class ETFTopHoldings:
    ticker: str
    holdings: tuple[ETFHoldingRow, ...]  # sorted weight desc, ticker asc tiebreak; capped at 10
    displayed_total: float  # sum of `holdings` weights (the displayed subset only)
    total_reference_count: int  # total constituents in the reference snapshot, before the 10-cap


def get_etf_top_holdings(ticker: str) -> ETFTopHoldings | None:
    """The single entry point the UI calls. Fails closed (returns None) for
    any non-ETF ticker (via the same canonical resolve_security_profile
    classification core.key_dates/core.analyst_view already use) or any ETF
    absent from ETF_HOLDINGS -- never a fabricated or inferred holdings
    list, never a fake-zero row. Deterministic and network-free: a pure
    dict lookup plus a stable sort, matching the exact reference data
    core.analytics already reads for Page 2 Look-through."""
    profile = resolve_security_profile(ticker)
    if profile is None or profile.kind_label != "ETF":
        return None
    constituents = ETF_HOLDINGS.get(profile.ticker)
    if not constituents:
        return None
    ranked = sorted(constituents.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:_MAX_HOLDINGS]
    rows = tuple(
        ETFHoldingRow(constituent_ticker, stock_name_en(constituent_ticker), weight)
        for constituent_ticker, weight in top
    )
    return ETFTopHoldings(
        profile.ticker, rows, sum(weight for _, weight in top), len(constituents),
    )
