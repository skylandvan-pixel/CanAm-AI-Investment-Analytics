"""Step 2A: deterministic, local-only Trade Impact Preview.

Converts an already-approved AI Action Plan recommendation ("reduce
approximately X% of the current NVDA position") into a concrete local
preview -- estimated whole shares, trade value, and before/after
exposure/concentration -- using only canonical portfolio data already held
client-side. The AI never performs this arithmetic and never receives
quantity, average cost, share counts, or realized gain/loss (see
core.analytics.canonical_fact_packet); this module is the only place those
numbers are computed, entirely after the fact, on data that already lives
in the user's own session.

Estimated realized gain/loss is included in the output shape but is
currently ALWAYS reported unavailable: HoldingInput.average_cost has no
guaranteed currency relationship to quote.price anywhere in the current
data model (manual entry and CSV/XLSX import never capture a cost-basis
currency), so `trade_value - shares * average_cost` cannot be proven to be
a same-currency subtraction. See _average_cost_currency_confirmed for the
single, explicit override point -- never inferred from account currency or
from holding.currency happening to match quote.currency.

Proceeds are always treated as held in cash (no destination/reinvestment
modeling -- that is explicitly out of scope for Step 2A). The "after" state
is produced by re-running the existing build_snapshot/analyze pipeline on a
hypothetical copy of the holdings, never by duplicating that logic here, so
identified look-through exposure and concentration metrics stay exactly
consistent with the real analysis (including the Step 1 lower-bound
semantics) with zero new market-data calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import floor

from core.analytics import analyze, build_snapshot
from core.models import AnalyticsResult, HoldingInput, Quote


@dataclass(frozen=True)
class TradeImpactPreview:
    ticker: str
    requested_position_reduction_pct: float
    executable_shares: float
    executable_position_reduction_pct: float
    trade_value: float
    trade_currency: str
    estimated_realized_gain_loss: float | None
    gain_loss_available: bool
    before_direct_weight: float | None
    after_direct_weight: float | None
    before_identified_exposure: float | None
    after_identified_exposure: float | None
    before_top5_concentration: float
    after_top5_concentration: float
    before_direct_effective_n: float | None
    after_direct_effective_n: float | None
    before_lookthrough_effective_n: float | None
    after_lookthrough_effective_n: float | None


def _direct_weight(result: AnalyticsResult, ticker: str) -> float | None:
    return next((p.weight for p in result.snapshot.complete_positions if p.ticker == ticker), None)


def _identified_exposure(result: AnalyticsResult, ticker: str) -> float | None:
    return next((x.true for x in result.true_exposures if x.ticker == ticker), None)


def _average_cost_currency_confirmed(target: HoldingInput, quote: Quote) -> bool:
    """Whether the current data model can GUARANTEE that average_cost is
    expressed in the same per-share currency as quote.price. It cannot, for
    any holding, today: neither manual entry nor CSV/XLSX import captures a
    cost-basis currency at all (see core/importers.py and
    app._holdings_from_frame), and HoldingInput.currency -- even when set --
    only designates the security's trading currency for valuation/FX
    purposes; it makes no claim about what currency the user typed
    average_cost in. This is the single, explicit override point for a
    future step that adds a genuine cost-basis-currency invariant -- never
    infer one from account currency, and never treat holding.currency
    matching quote.currency as proof, since neither establishes anything
    about average_cost specifically."""
    return False


def _average_cost_is_usable(average_cost: float | None) -> bool:
    """A finite, positive number -- rejects None, <=0, NaN, and +/-inf so a
    corrupted or nonsensical stored value can never propagate into
    -inf/NaN arithmetic."""
    return average_cost is not None and math.isfinite(average_cost) and average_cost > 0


def build_trade_impact_preview(
    *, holdings: list[HoldingInput], quotes: dict[str, Quote], cash: float,
    account_currency: str, usd_cad: float | None, account_type: str,
    before_result: AnalyticsResult, ticker: str, action: str,
    position_reduction_pct: float | None,
) -> TradeImpactPreview | None:
    """Returns None whenever the preview cannot be produced safely -- the
    caller must then show nothing (never fabricate a partial preview). This
    is the single, complete suppression gate: action must be "REDUCE" (an
    immediate proposed reduction) or "REDUCE_ON_REBOUND" (a conditional
    reduction sized the same way, to execute IF/WHEN its trigger is
    satisfied -- the caller is responsible for labeling that preview as
    conditional, since this function's deterministic math is identical
    either way), position_reduction_pct must be a usable (0, 100)
    percentage, the ticker must be a real direct holding with an available
    price, and the requested reduction must not round down to zero
    executable shares -- any one of these failing means no preview, never a
    partial one. STAGED_SELL and every other action remain ineligible
    (Step 2A.2 deliberately does not extend this to an open-ended
    multi-tranche plan)."""
    if action not in ("REDUCE", "REDUCE_ON_REBOUND") or position_reduction_pct is None:
        return None
    if not (0 < position_reduction_pct < 100):
        return None

    ticker = ticker.strip().upper().replace(".", "-")
    try:
        normalized = [h.normalized() for h in holdings]
    except ValueError:
        # An invalid holding anywhere in the portfolio (e.g. a corrupted
        # quantity) must suppress the preview, never crash the Action Plan.
        # The canonical portfolio validation this mirrors lives in
        # HoldingInput.normalized() itself -- unchanged, not re-implemented.
        return None
    target = next((h for h in normalized if h.ticker == ticker), None)
    if target is None:
        return None

    quote = quotes.get(ticker)
    if quote is None or quote.price is None or quote.status.startswith(("error", "identity:")):
        return None

    executable_shares = floor(target.quantity * (position_reduction_pct / 100))
    executable_shares = min(executable_shares, floor(target.quantity))
    if executable_shares <= 0:
        return None

    trade_currency = target.currency or quote.currency or account_currency
    trade_value = executable_shares * quote.price
    gain_loss_available = (
        _average_cost_is_usable(target.average_cost)
        and _average_cost_currency_confirmed(target, quote)
    )
    estimated_realized_gain_loss = (
        trade_value - executable_shares * target.average_cost if gain_loss_available else None
    )

    # Proceeds -> cash only (Step 2A scope: no reinvestment/destination
    # modeling). Convert into account-currency terms using the same FX rate
    # already resolved for the real analysis -- never a second FX lookup.
    proceeds_in_account_ccy = trade_value
    if trade_currency != account_currency:
        if usd_cad is None or usd_cad <= 0:
            return None
        proceeds_in_account_ccy = trade_value * usd_cad if trade_currency == "USD" else trade_value / usd_cad

    hypothetical_holdings = [
        HoldingInput(
            h.ticker, h.quantity - executable_shares if h.ticker == ticker else h.quantity,
            h.average_cost, h.currency,
        )
        for h in normalized
    ]
    hypothetical_holdings = [h for h in hypothetical_holdings if h.quantity > 0]

    after_result: AnalyticsResult | None = None
    if hypothetical_holdings:
        after_snapshot = build_snapshot(
            hypothetical_holdings, quotes, cash=cash + proceeds_in_account_ccy,
            account_currency=account_currency, usd_cad=usd_cad, account_type=account_type,
        )
        after_result = analyze(after_snapshot)

    return TradeImpactPreview(
        ticker=ticker,
        requested_position_reduction_pct=position_reduction_pct,
        executable_shares=executable_shares,
        executable_position_reduction_pct=round(executable_shares / target.quantity * 100, 2),
        trade_value=trade_value,
        trade_currency=trade_currency,
        estimated_realized_gain_loss=estimated_realized_gain_loss,
        gain_loss_available=gain_loss_available,
        before_direct_weight=_direct_weight(before_result, ticker),
        after_direct_weight=_direct_weight(after_result, ticker) if after_result else 0.0,
        before_identified_exposure=_identified_exposure(before_result, ticker),
        after_identified_exposure=_identified_exposure(after_result, ticker) if after_result else 0.0,
        before_top5_concentration=before_result.top5_concentration,
        after_top5_concentration=after_result.top5_concentration if after_result else 0.0,
        before_direct_effective_n=before_result.direct_effective_n,
        after_direct_effective_n=after_result.direct_effective_n if after_result else None,
        before_lookthrough_effective_n=before_result.lookthrough_effective_n,
        after_lookthrough_effective_n=after_result.lookthrough_effective_n if after_result else None,
    )
