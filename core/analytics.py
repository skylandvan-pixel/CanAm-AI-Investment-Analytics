from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

from core.models import (
    AnalyticsResult, Exposure, HoldingInput, PortfolioSnapshot, Position, Quote, RiskFlag, normalize_account_type,
)
from core.reference import ASSET_CLASS_LABELS_EN, ASSET_CLASS_LABELS_ZH, CASH_LIKE_TICKERS, ETF_HOLDINGS, ETF_META, STOCK_SECTORS


def build_snapshot(
    holdings: list[HoldingInput], quotes: dict[str, Quote], *, cash: float = 0.0,
    account_currency: str = "USD", usd_cad: float | None = None, account_type: str = "Taxable",
) -> PortfolioSnapshot:
    account_currency = account_currency.upper()
    if account_currency not in {"USD", "CAD"} or cash < 0:
        raise ValueError("Account currency must be USD/CAD and cash cannot be negative")
    normalized = [h.normalized() for h in holdings]
    rows: list[dict] = []
    for holding in normalized:
        quote = quotes.get(holding.ticker)
        issues: list[str] = []
        value = None
        change = None
        price = quote.price if quote else None
        security_ccy = holding.currency or (quote.currency if quote else None)
        if quote is not None and quote.status.startswith("identity:"):
            # A resolved-but-mismatched or unresolved ticker identity: the
            # exact fail-closed message is carried in the quote status by
            # core.market.fetch_quotes / core.ticker_resolution so it can
            # surface unmodified to the user instead of a generic notice.
            issues.append(quote.status.split("identity:", 1)[1])
        elif quote is None or quote.price is None or quote.status.startswith("error"):
            issues.append("Verified market price unavailable")
        elif security_ccy != account_currency:
            if usd_cad is None or usd_cad <= 0:
                issues.append("Verified USD/CAD rate unavailable")
            elif security_ccy == "USD" and account_currency == "CAD":
                value = holding.quantity * quote.price * usd_cad
            elif security_ccy == "CAD" and account_currency == "USD":
                value = holding.quantity * quote.price / usd_cad
        else:
            value = holding.quantity * quote.price
        if quote and quote.price and quote.previous_close and quote.previous_close > 0:
            change = quote.price / quote.previous_close - 1
        rows.append(dict(holding=holding, quote=quote, value=value, change=change, issues=issues, security_ccy=security_ccy))

    total = cash + sum(row["value"] or 0 for row in rows)
    if total <= 0:
        raise ValueError("Portfolio has no verified positive value")
    positions = tuple(
        Position(
            # The resolved security currency (falls back to the quote's real
            # currency when the holding doesn't specify one), not the raw,
            # possibly-unset HoldingInput.currency -- downstream FX/foreign-
            # exposure scoring keys off this field.
            row["holding"].ticker, row["holding"].quantity, row["security_ccy"] or account_currency,
            row["quote"].price if row["quote"] else None, row["value"],
            row["value"] / total if row["value"] is not None else None,
            row["change"], "complete" if row["value"] is not None else "incomplete",
            tuple(row["issues"]),
        ) for row in rows
    )
    # A fully missing price has no trustworthy denominator for a value-
    # weighted coverage ratio. Preserve the legacy system's fail-closed
    # data-quality semantics by reporting priced-position coverage instead
    # of silently treating the known subset as 100% of the portfolio.
    coverage = (sum(p.status == "complete" for p in positions) / len(positions)) if positions else 1.0
    return PortfolioSnapshot(account_currency, cash, positions, total, cash / total, coverage, account_type=normalize_account_type(account_type))


def _effective_n(weights: list[float]) -> float | None:
    denominator = sum(w * w for w in weights if w > 0)
    return 1 / denominator if denominator else None


def _score(snapshot: PortfolioSnapshot, allocation: dict[str, float], sectors: dict[str, float], top3: float, top5: float) -> float:
    positions = snapshot.complete_positions
    stock_weights = [p.weight or 0 for p in positions if p.ticker not in ETF_META]
    largest_stock = max(stock_weights, default=0.0)
    div = 15 * (1 - min(1, max(0, (top5 - .40) / .60))) if top5 > .40 else 15
    if len(positions) < 3:
        div = min(div, 7.5)
    conc = 15 - 15 * min(1, largest_stock / .30) * .6 - 15 * min(1, max(0, (top3 - .30) / .50)) * .4
    liquidity = snapshot.cash_weight + sum((p.weight or 0) for p in positions if p.ticker in CASH_LIKE_TICKERS)
    liq = 10 if .05 <= liquidity <= .25 else 10 * (1 - min(1, (.05 - liquidity if liquidity < .05 else liquidity - .25) / .30))
    equity = allocation.get("Equity", 0)
    alloc = 15 if .55 <= equity <= .75 else 15 * (1 - min(1, (.55 - equity if equity < .55 else equity - .75) / .30))
    largest_sector = max(sectors.values(), default=0)
    category = 15 * (1 - min(1, max(0, (largest_sector - .30) / .50))) if largest_sector > .30 else 15
    foreign = sum((p.weight or 0) for p in positions if p.currency != snapshot.account_currency)
    fx = 10 * (1 - min(1, max(0, (foreign - .60) / .40))) if foreign > .60 else 10
    points = sum(max(0, x) for x in (div, conc, liq, alloc, category, fx))
    raw = Decimal(str(points)) * Decimal("100") / Decimal("80")
    return float(raw.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def analyze(snapshot: PortfolioSnapshot) -> AnalyticsResult:
    positions = list(snapshot.complete_positions)
    direct = {p.ticker: p.weight or 0 for p in positions}
    ranked = sorted(direct.items(), key=lambda item: item[1], reverse=True)
    top3, top5 = sum(w for _, w in ranked[:3]), sum(w for _, w in ranked[:5])

    allocation: dict[str, float] = defaultdict(float)
    sectors: dict[str, float] = defaultdict(float)
    true_weights: dict[str, float] = defaultdict(float)
    indirect: dict[str, float] = defaultdict(float)
    lookthrough_vector: dict[str, float] = defaultdict(float)
    uncovered: list[str] = []

    if snapshot.cash_weight:
        allocation["Cash"] += snapshot.cash_weight
        lookthrough_vector["CASH"] += snapshot.cash_weight
    for p in positions:
        weight = p.weight or 0
        if p.ticker in ETF_META:
            sector, asset = ETF_META[p.ticker]
            allocation[asset] += weight
            sectors[sector] += weight
            if asset == "Equity":
                constituents = ETF_HOLDINGS.get(p.ticker)
                if constituents is None:
                    uncovered.append(p.ticker)
                    lookthrough_vector[p.ticker] += weight
                else:
                    named = min(1.0, sum(constituents.values()))
                    for ticker, fraction in constituents.items():
                        contribution = weight * fraction
                        indirect[ticker] += contribution
                        true_weights[ticker] += contribution
                        lookthrough_vector[ticker] += contribution
                    residual = weight * (1 - named)
                    if residual > 0:
                        lookthrough_vector[p.ticker] += residual
            else:
                lookthrough_vector[p.ticker] += weight
        else:
            allocation["Equity"] += weight
            sectors[STOCK_SECTORS.get(p.ticker, "Other")] += weight
            true_weights[p.ticker] += weight
            lookthrough_vector[p.ticker] += weight

    exposures = tuple(sorted(
        (Exposure(t, direct.get(t, 0), indirect.get(t, 0), w) for t, w in true_weights.items() if w > 0),
        key=lambda x: x.true, reverse=True,
    ))
    covered_equity_etf_weight = sum(direct[t] for t in ETF_HOLDINGS if t in direct)
    uncovered_weight = sum(direct[t] for t in uncovered)
    coverage = "complete" if not uncovered else ("partial" if uncovered_weight < .15 else "insufficient")
    direct_n = _effective_n([p.weight or 0 for p in positions] + ([snapshot.cash_weight] if snapshot.cash_weight else []))
    lookthrough_n = _effective_n(list(lookthrough_vector.values()))

    technology = sectors.get("Technology", 0)
    flags: list[RiskFlag] = []
    largest_true = exposures[0] if exposures else None
    if largest_true and largest_true.true > .15:
        flags.append(RiskFlag("single_name", "单一标的集中度过高（Single-Name Concentration）", f"{largest_true.ticker} 是当前组合最主要的单一股票集中风险。", "high"))
    elif largest_true and largest_true.true > .10:
        flags.append(RiskFlag("single_name", "单一标的集中度（Single-Name Concentration）", f"{largest_true.ticker} 是当前组合中集中度最高的单一持仓，建议持续关注。"))
    if technology > .40:
        flags.append(RiskFlag("sector", "科技板块集中度过高（Technology Concentration）", f"直接持仓中科技板块占比为 {technology:.1%}。", "high"))
    elif technology > .25:
        flags.append(RiskFlag("sector", "科技板块集中度（Technology Concentration）", f"直接持仓中科技板块占比为 {technology:.1%}。"))
    liquidity = snapshot.cash_weight + sum(direct.get(t, 0) for t in CASH_LIKE_TICKERS)
    if liquidity < .05:
        flags.append(RiskFlag("liquidity", "现金储备偏低（Limited Liquidity Reserve）", f"现金及类现金储备占比为 {liquidity:.1%}。"))
    flags = flags[:3]
    risk_level = "High" if any(f.severity == "high" for f in flags) else ("Elevated" if flags else "Measured")
    score = _score(snapshot, allocation, sectors, top3, top5)
    top_asset = max(allocation, key=allocation.get)
    summary = f"组合以{ASSET_CLASS_LABELS_ZH.get(top_asset, top_asset)}（{ASSET_CLASS_LABELS_EN.get(top_asset, top_asset)}）为主，前三大直接持仓合计 {top3:.1%}。"
    canonical = {
        "direct_allocation": "overview.treemap", "asset_allocation": "overview.asset_allocation",
        "portfolio_score": "overview.summary", "direct_concentration": "risk.concentration",
        "etf_lookthrough": "risk.lookthrough", "true_exposure": "risk.true_exposure",
        "risk_flags": "risk.flags", "committee": "ai.committee",
        "chairman": "ai.chairman", "action_plan": "ai.action_plan",
    }
    return AnalyticsResult(
        snapshot, score, risk_level, dict(allocation), dict(sectors), tuple(ranked[:8]), exposures,
        top3, top5, direct_n, lookthrough_n, coverage, tuple(sorted(uncovered)), tuple(flags),
        summary, canonical,
        {"valuation_coverage": snapshot.coverage_ratio, "lookthrough_covered_weight": covered_equity_etf_weight,
         "lookthrough_uncovered_weight": uncovered_weight, "lookthrough_reference": "Top-N as of 2025-06-30"},
    )


def treemap_rows(result: AnalyticsResult, *, minimum_weight: float = .025) -> list[dict]:
    """Direct weights only; small positions are aggregated to Other."""
    rows, other = [], 0.0
    by_ticker = {p.ticker: p for p in result.snapshot.complete_positions}
    for ticker, weight in sorted(((p.ticker, p.weight or 0) for p in result.snapshot.complete_positions), key=lambda x: x[1], reverse=True):
        if weight < minimum_weight:
            other += weight
            continue
        p = by_ticker[ticker]
        rows.append({"ticker": ticker, "weight": weight, "daily_change": p.daily_change})
    if other > 0:
        rows.append({"ticker": "Other", "weight": other, "daily_change": None})
    return rows


def canonical_fact_packet(result: AnalyticsResult) -> dict:
    def clean(value: float | None):
        return round(value, 6) if value is not None and isfinite(value) else None
    return {
        "portfolio": {"account_currency": result.snapshot.account_currency, "account_type": result.snapshot.account_type,
                      "portfolio_score": result.portfolio_score,
                      "risk_level": result.risk_level, "total_assets": round(result.snapshot.total_assets, 2)},
        "asset_allocation": {k: clean(v) for k, v in result.asset_allocation.items()},
        "top_direct_holdings": [{"ticker": t, "weight": clean(w)} for t, w in result.top_direct[:5]],
        "top_true_exposures": [{"ticker": x.ticker, "direct": clean(x.direct), "indirect": clean(x.indirect), "true": clean(x.true)} for x in result.true_exposures[:8]],
        "concentration": {"top3": clean(result.top3_concentration), "top5": clean(result.top5_concentration),
                          "direct_effective_n": clean(result.direct_effective_n), "lookthrough_effective_n": clean(result.lookthrough_effective_n)},
        "risk_flags": [{"key": f.key, "title": f.title, "detail": f.detail} for f in result.risk_flags],
        "data_quality": result.data_quality,
    }
