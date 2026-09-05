from __future__ import annotations

import pytest

from core.analytics import analyze, build_snapshot, canonical_fact_packet
from core.models import HoldingInput, Quote
from core.trade_preview import TradeImpactPreview, build_trade_impact_preview


def _preview(holdings, quotes, *, cash=1000.0, account_currency="USD", usd_cad=None,
             account_type="Taxable", ticker, action="REDUCE", position_reduction_pct):
    before = analyze(build_snapshot(holdings, quotes, cash=cash, account_currency=account_currency,
                                     usd_cad=usd_cad, account_type=account_type))
    return build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=cash, account_currency=account_currency,
        usd_cad=usd_cad, account_type=account_type, before_result=before, ticker=ticker,
        action=action, position_reduction_pct=position_reduction_pct,
    )


@pytest.fixture
def synthetic_two_stock():
    """Two plain (non-ETF) synthetic tickers -- deliberately not NVDA/VOO, so
    rounding/cash/gain-loss behavior isn't overfit to the demo portfolio."""
    holdings = [HoldingInput("ACME", 143, average_cost=80.0), HoldingInput("WIDGE", 50, average_cost=40.0)]
    quotes = {
        "ACME": Quote("ACME", 120.0, "USD", 118.0, "test", "verified"),
        "WIDGE": Quote("WIDGE", 60.0, "USD", 59.0, "test", "verified"),
    }
    return holdings, quotes


def test_ten_percent_reduction_floor_rounds_shares(synthetic_two_stock):
    """A: 143 shares * 10% = 14.3 -> floors to 14, never rounds up."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview.executable_shares == 14
    assert preview.requested_position_reduction_pct == 10.0
    assert preview.executable_position_reduction_pct == pytest.approx(14 / 143 * 100, abs=0.01)


def test_executable_shares_never_exceed_current_quantity():
    """B: even a reduction pct very close to 100 must never sell more than
    the held quantity, for a small integer position."""
    holdings = [HoldingInput("ACME", 3, average_cost=10.0)]
    quotes = {"ACME": Quote("ACME", 50.0, "USD", 49.0, "test", "verified")}
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=99.0)
    assert preview.executable_shares <= 3


def test_very_small_reduction_rounding_to_zero_is_non_actionable():
    """C: 1 share * 10% = 0.1 -> floors to 0 -> no preview at all."""
    holdings = [HoldingInput("ACME", 1, average_cost=10.0)]
    quotes = {"ACME": Quote("ACME", 50.0, "USD", 49.0, "test", "verified")}
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview is None


def test_trade_value_equals_shares_times_current_price(synthetic_two_stock):
    """D."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview.trade_value == pytest.approx(preview.executable_shares * 120.0)


def test_realized_gain_loss_is_currently_unavailable_pending_cost_currency_guarantee(synthetic_two_stock):
    """E (revised per Codex P0 fix): even with a usable, finite, positive
    average_cost, the estimate must stay unavailable, because nothing in the
    current data model guarantees average_cost is expressed in the same
    currency as quote.price (see core.trade_preview.
    _average_cost_currency_confirmed). The rest of the preview (shares,
    trade value, before/after metrics) still renders normally."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview is not None
    assert preview.gain_loss_available is False
    assert preview.estimated_realized_gain_loss is None
    assert preview.executable_shares == 14
    assert preview.trade_value == pytest.approx(14 * 120.0)


def test_missing_average_cost_does_not_fabricate_gain_loss():
    """F: no average_cost -> gain/loss must be reported as unavailable, not
    guessed as zero or omitted silently -- the rest of the preview still
    renders (non-tax trade-impact numbers)."""
    holdings = [HoldingInput("ACME", 100)]  # average_cost defaults to None
    quotes = {"ACME": Quote("ACME", 120.0, "USD", 118.0, "test", "verified")}
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview is not None
    assert preview.gain_loss_available is False
    assert preview.estimated_realized_gain_loss is None
    assert preview.executable_shares == 10
    assert preview.trade_value == pytest.approx(1200.0)


def test_hypothetical_portfolio_does_not_mutate_original_holdings(synthetic_two_stock):
    """G: the input holdings list/objects must be unchanged after the call
    (HoldingInput is frozen, but assert the caller's own list/values too)."""
    holdings, quotes = synthetic_two_stock
    original_quantities = [h.quantity for h in holdings]
    _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert [h.quantity for h in holdings] == original_quantities


def test_sale_proceeds_increase_hypothetical_cash_and_total_value_is_constant(synthetic_two_stock):
    """H + I: reconstruct the after-state the same way the module does and
    confirm cash rose by exactly the trade value (same-currency case) while
    total portfolio value stays constant (no commissions modeled)."""
    holdings, quotes = synthetic_two_stock
    before = analyze(build_snapshot(holdings, quotes, cash=1000.0, account_currency="USD", account_type="Taxable"))
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=1000.0, account_currency="USD", usd_cad=None,
        account_type="Taxable", before_result=before, ticker="ACME", action="REDUCE",
        position_reduction_pct=10.0,
    )
    after_holdings = [HoldingInput("ACME", 143 - preview.executable_shares, 80.0), HoldingInput("WIDGE", 50, 40.0)]
    after_cash = 1000.0 + preview.trade_value
    after_snapshot = build_snapshot(after_holdings, quotes, cash=after_cash, account_currency="USD", account_type="Taxable")
    assert after_snapshot.cash == pytest.approx(1000.0 + preview.trade_value)
    assert after_snapshot.total_assets == pytest.approx(before.snapshot.total_assets)


def test_before_after_direct_weight_changes_correctly(synthetic_two_stock):
    """J: direct weight of the reduced ticker must strictly decrease."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview.before_direct_weight > preview.after_direct_weight


def test_identified_exposure_declines_only_by_direct_component_via_real_etf_lookthrough():
    """K + L: use NVDA (a real ETF_HOLDINGS constituent of VOO) so the
    scenario exercises genuine indirect exposure -- reducing NVDA's DIRECT
    holding must lower NVDA's identified true exposure by exactly the same
    amount as its direct weight drop, leaving VOO's indirect NVDA
    contribution (and hence VOO's own reported weight) untouched."""
    holdings = [HoldingInput("NVDA", 100, average_cost=50.0), HoldingInput("VOO", 50, average_cost=300.0)]
    quotes = {
        "NVDA": Quote("NVDA", 100.0, "USD", 99.0, "test", "verified"),
        "VOO": Quote("VOO", 400.0, "USD", 398.0, "test", "verified"),
    }
    before = analyze(build_snapshot(holdings, quotes, cash=500.0, account_currency="USD", account_type="Taxable"))
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=500.0, account_currency="USD", usd_cad=None,
        account_type="Taxable", before_result=before, ticker="NVDA", action="REDUCE",
        position_reduction_pct=20.0,
    )
    assert preview is not None
    direct_drop = preview.before_direct_weight - preview.after_direct_weight
    exposure_drop = preview.before_identified_exposure - preview.after_identified_exposure
    assert exposure_drop == pytest.approx(direct_drop, abs=1e-9)

    # VOO itself (and therefore its untouched indirect NVDA contribution) is
    # unaffected -- only cash and NVDA's own quantity changed.
    after_holdings = [HoldingInput("NVDA", 100 - preview.executable_shares, 50.0), HoldingInput("VOO", 50, 300.0)]
    after_result = analyze(build_snapshot(after_holdings, quotes, cash=500.0 + preview.trade_value,
                                           account_currency="USD", account_type="Taxable"))
    voo_before = next(w for t, w in before.top_direct if t == "VOO")
    voo_after = next(w for t, w in after_result.top_direct if t == "VOO")
    assert voo_before == pytest.approx(voo_after)


def test_incomplete_lookthrough_true_value_is_reused_unmodified_not_recomputed():
    """M: trade_preview must never adjust/soften/re-derive the true-exposure
    number itself -- it reuses analyze()'s own Exposure.true value verbatim
    for both before and after, so it automatically inherits whatever
    lower-bound/identified-exposure semantics Step 1 established (an
    uncovered ETF's residual weight is never attributed to a specific
    stock -- see core/analytics.py), without this module re-deriving or
    weakening that guarantee."""
    holdings = [HoldingInput("NVDA", 100, average_cost=50.0), HoldingInput("VXUS", 80, average_cost=55.0)]
    quotes = {
        "NVDA": Quote("NVDA", 100.0, "USD", 99.0, "test", "verified"),
        "VXUS": Quote("VXUS", 60.0, "USD", 59.0, "test", "verified"),  # VXUS: no ETF_HOLDINGS entry (uncovered)
    }
    before = analyze(build_snapshot(holdings, quotes, cash=200.0, account_currency="USD", account_type="Taxable"))
    assert before.lookthrough_coverage in {"partial", "insufficient"}  # VXUS is genuinely uncovered
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=200.0, account_currency="USD", usd_cad=None,
        account_type="Taxable", before_result=before, ticker="NVDA", action="REDUCE",
        position_reduction_pct=25.0,
    )
    assert preview is not None
    before_true = next(x.true for x in before.true_exposures if x.ticker == "NVDA")
    assert preview.before_identified_exposure == before_true  # exact reuse, not a re-derived/adjusted figure


def test_privacy_invariant_quantity_and_average_cost_absent_from_fact_packet(synthetic_two_stock):
    """N: the deterministic preview computes locally from quantity/average
    cost, but none of that (nor the realized gain/loss it derives) may ever
    reach the AI fact packet."""
    holdings, quotes = synthetic_two_stock
    result = analyze(build_snapshot(holdings, quotes, cash=1000.0, account_currency="USD", account_type="Taxable"))
    packet = canonical_fact_packet(result)
    packet_text = str(packet)
    assert "quantity" not in packet_text
    assert "average_cost" not in packet_text
    assert "143" not in packet_text  # the raw share count itself
    assert "80.0" not in packet_text  # the raw average cost itself


@pytest.mark.parametrize("account_type", ["TFSA", "Taxable"])
def test_tfsa_and_taxable_both_support_the_deterministic_preview(account_type, synthetic_two_stock):
    """O: the preview is a pure local calculation, indifferent to account
    type -- it must work identically for both (Step 2A adds no tax logic of
    any kind for either account type)."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0, account_type=account_type)
    assert preview is not None
    assert preview.executable_shares == 14


def test_no_tax_estimate_fields_exist_on_the_preview_model():
    """P: TradeImpactPreview must expose only an estimate of realized
    gain/loss on the sale itself -- never a tax payable/owing/rate field."""
    fields = set(TradeImpactPreview.__dataclass_fields__)
    for forbidden in ("tax_payable", "tax_owing", "tax_rate", "marginal_tax", "tax_savings", "cra"):
        assert forbidden not in fields


def test_no_tax_loss_harvesting_fields_or_helpers_exist():
    """Q: Step 2A must not introduce any loss-harvesting-candidate search,
    offset matching, or tax-lot optimization surface."""
    import core.trade_preview as module

    fields = set(TradeImpactPreview.__dataclass_fields__)
    for forbidden in ("harvest", "loss_candidate", "offset", "tax_lot", "superficial_loss"):
        assert forbidden not in fields
        assert not any(forbidden in name.lower() for name in dir(module) if not name.startswith("_"))


def test_invalid_ticker_suppresses_preview(synthetic_two_stock):
    """R."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="NOTHELD", position_reduction_pct=10.0)
    assert preview is None


def test_non_reduce_action_suppresses_preview(synthetic_two_stock):
    """S."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", action="HOLD", position_reduction_pct=10.0)
    assert preview is None
    preview_none_action = _preview(holdings, quotes, ticker="ACME", action="REDUCE_ON_REBOUND", position_reduction_pct=10.0)
    assert preview_none_action is None


@pytest.mark.parametrize("pct", [100.0, 150.0])
def test_reduction_at_or_above_100_percent_fails_safely(synthetic_two_stock, pct):
    """T."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=pct)
    assert preview is None


def test_reduction_pct_none_suppresses_preview(synthetic_two_stock):
    """Companion to S/T: the AI declining to give a specific percentage
    (position_reduction_pct left null) must suppress the preview, never
    fall back to guessing one."""
    holdings, quotes = synthetic_two_stock
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=None)
    assert preview is None


def test_zero_or_negative_reduction_pct_suppresses_preview(synthetic_two_stock):
    holdings, quotes = synthetic_two_stock
    assert _preview(holdings, quotes, ticker="ACME", position_reduction_pct=0.0) is None
    assert _preview(holdings, quotes, ticker="ACME", position_reduction_pct=-5.0) is None


def test_unpriced_ticker_suppresses_preview():
    holdings = [HoldingInput("ACME", 100, average_cost=10.0)]
    quotes = {"ACME": Quote("ACME", None, None, status="error:test")}
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview is None


# ---------------------------------------------------------------------------
# Codex pre-commit fixes (P0 average-cost currency safety, non-finite
# average_cost, invalid-quantity fail-safe, cross-currency regression).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), 0.0])
def test_non_finite_or_zero_average_cost_does_not_fabricate_gain_loss(bad_cost):
    """Fix 2: NaN/+inf/0 average_cost must never propagate into arithmetic
    (e.g. NaN/inf gain/loss) -- treated like a missing cost basis:
    unavailable, not fabricated, never a crash. These three values are not
    already rejected by HoldingInput.normalized()'s own `< 0` check (NaN and
    +inf both compare False to `< 0`; 0 is not negative), so they reach this
    module's own guard -- the rest of the preview still renders."""
    from core.trade_preview import _average_cost_is_usable

    assert _average_cost_is_usable(bad_cost) is False  # the guard itself, tested directly

    holdings = [HoldingInput("ACME", 100, average_cost=bad_cost)]
    quotes = {"ACME": Quote("ACME", 120.0, "USD", 118.0, "test", "verified")}
    preview = _preview(holdings, quotes, ticker="ACME", position_reduction_pct=10.0)
    assert preview is not None  # the non-tax part of the preview still renders
    assert preview.gain_loss_available is False
    assert preview.estimated_realized_gain_loss is None


@pytest.mark.parametrize("bad_cost", [-5.0, float("-inf")])
def test_negative_average_cost_is_already_rejected_upstream_and_suppresses_preview(bad_cost):
    """Companion: a negative (or -inf) average_cost is already rejected by
    the existing, unmodified HoldingInput.normalized() at snapshot-build
    time -- Fix 4's try/except around normalization means that surfaces as a
    suppressed preview (None), not a crash, one layer up from Fix 2's own
    finiteness guard."""
    holdings = [HoldingInput("ACME", 100, average_cost=bad_cost)]
    quotes = {"ACME": Quote("ACME", 120.0, "USD", 118.0, "test", "verified")}
    before = analyze(build_snapshot([HoldingInput("WIDGE", 10)], quotes, cash=1000.0,
                                     account_currency="USD", account_type="Taxable"))
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=1000.0, account_currency="USD", usd_cad=None,
        account_type="Taxable", before_result=before, ticker="ACME", action="REDUCE",
        position_reduction_pct=10.0,
    )
    assert preview is None


def test_finite_positive_average_cost_passes_the_finiteness_guard_alone():
    """Companion: the finiteness guard itself accepts a normal value (the
    end-to-end result is still gain_loss_available=False today, because the
    separate currency-confirmation gate -- Fix 1 -- is unconditionally
    False; see test_realized_gain_loss_is_currently_unavailable_pending_
    cost_currency_guarantee)."""
    from core.trade_preview import _average_cost_is_usable

    assert _average_cost_is_usable(80.0) is True


def test_invalid_quantity_suppresses_preview_instead_of_raising():
    """Fix 4: a corrupted holding (invalid quantity) anywhere in the
    portfolio must suppress the preview, never propagate HoldingInput.
    normalized()'s ValueError up through the Action Plan rendering."""
    holdings = [HoldingInput("ACME", -5, average_cost=10.0)]  # invalid: quantity <= 0
    quotes = {"ACME": Quote("ACME", 50.0, "USD", 49.0, "test", "verified")}
    before = analyze(build_snapshot([HoldingInput("WIDGE", 10)], quotes, cash=1000.0,
                                     account_currency="USD", account_type="Taxable"))
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=1000.0, account_currency="USD", usd_cad=None,
        account_type="Taxable", before_result=before, ticker="ACME", action="REDUCE",
        position_reduction_pct=10.0,
    )
    assert preview is None


def test_cross_currency_cad_account_usd_security_reduce():
    """Fix 5: CAD account holding a USD-quoted security, REDUCE 20% of NVDA
    with VOO also held (real ETF look-through). Verifies: trade_value stays
    in the security's own USD; sale proceeds are converted into CAD using
    the existing usd_cad rate (never a second/new FX lookup); total
    hypothetical portfolio value is invariant; VOO's own reported weight
    (and hence its untouched indirect NVDA contribution) is unaffected;
    and -- per Fix 1 -- realized gain/loss stays unavailable regardless of
    currency, since no currency guarantee exists for average_cost at all."""
    holdings = [HoldingInput("NVDA", 100, average_cost=50.0), HoldingInput("VOO", 50, average_cost=300.0)]
    quotes = {
        "NVDA": Quote("NVDA", 100.0, "USD", 99.0, "test", "verified"),
        "VOO": Quote("VOO", 400.0, "USD", 398.0, "test", "verified"),
    }
    usd_cad = 1.35
    before = analyze(build_snapshot(holdings, quotes, cash=500.0, account_currency="CAD",
                                     usd_cad=usd_cad, account_type="Taxable"))
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=500.0, account_currency="CAD", usd_cad=usd_cad,
        account_type="Taxable", before_result=before, ticker="NVDA", action="REDUCE",
        position_reduction_pct=20.0,
    )
    assert preview is not None
    assert preview.trade_currency == "USD"
    assert preview.gain_loss_available is False
    assert preview.estimated_realized_gain_loss is None

    expected_cad_proceeds = preview.trade_value * usd_cad
    after_holdings = [HoldingInput("NVDA", 100 - preview.executable_shares, 50.0), HoldingInput("VOO", 50, 300.0)]
    after_snapshot = build_snapshot(after_holdings, quotes, cash=500.0 + expected_cad_proceeds,
                                     account_currency="CAD", usd_cad=usd_cad, account_type="Taxable")
    after_result = analyze(after_snapshot)

    assert after_snapshot.cash == pytest.approx(500.0 + expected_cad_proceeds)
    assert after_snapshot.total_assets == pytest.approx(before.snapshot.total_assets)
    assert preview.after_direct_weight == pytest.approx(next(w for t, w in after_result.top_direct if t == "NVDA"))

    voo_before = next(w for t, w in before.top_direct if t == "VOO")
    voo_after = next(w for t, w in after_result.top_direct if t == "VOO")
    assert voo_before == pytest.approx(voo_after)  # ETF indirect exposure invariant
