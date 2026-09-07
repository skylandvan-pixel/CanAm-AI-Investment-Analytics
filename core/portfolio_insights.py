"""Step 2A.6 -- Deterministic Portfolio Insights.

Replaces the old low-value Page 1 summary sentence ("组合以股票（Equity）为主，
前三大直接持仓合计 XX%。") with at most two short, interpretive insights built
entirely from AnalyticsResult -- the same canonical Layer 1 output already
computed by core.analytics.analyze(). No new calculation, no new financial
model, no LLM call: every number used here is read directly off an existing
field (risk_flags, true_exposures, asset_allocation, or direct_effective_n),
never recomputed with new logic.

Selection priority for the first ("dynamic") slot:
  A. a single_name or sector risk_flag exists -> concentration insight
  D. else a direct-holdings diversification statement, quantified with
     direct_effective_n when available

A liquidity-only flag set (no single_name, no sector) deliberately produces
no dynamic-slot insight at all and falls through to D -- see the Step 2A.6
Final Semantic Fix Pass report: liquidity is a distinct risk dimension from
concentration and must never be worded as one.

There is deliberately NO currency-exposure branch: Position.currency is the
resolved quote/settlement currency (the same field core.analytics._score()
sums for its own FX scoring term), not a canonical economic-currency-
exposure field -- a CAD-listed, USD-index-tracking ETF like ZSP would be
wrongly counted as "CAD exposure" under a settlement-currency rule, which is
exactly the false economic-exposure inference this module must not make. No
such canonical field exists on AnalyticsResult, so Step 2A.6 has no
currency-insight branch at all.

The second ("structure") slot is always attempted from asset_allocation.
Either slot can come back None (fail-closed -- never a fabricated filler
insight); the final list is capped at 2 and can be empty.
"""

from __future__ import annotations

from core.models import AnalyticsResult, Exposure, RiskFlag

# A look-through-derived true exposure must exceed direct weight by at least
# this much before the look-through clause is worth citing -- avoids citing
# an indirect contribution too small to change the reader's interpretation.
_MATERIAL_INDIRECT_DELTA = 0.01


def _concentration_insight(risk_flags: tuple[RiskFlag, ...], true_exposures: tuple[Exposure, ...]) -> dict | None:
    """Priority A. Explicitly distinguishes flag KIND rather than trusting
    risk_flags[0] blindly -- only single_name or sector flags ever produce
    concentration language. A liquidity-only flag set (or no flags at all)
    returns None here and falls through to the diversification insight (D);
    liquidity is a distinct risk dimension and must never be worded as a
    concentration finding.

    single_name is enriched with the matching true_exposures[0] entry (the
    same Exposure the flag itself was derived from) to add the look-through
    dimension when it's material, always preserving "≥" lower-bound
    semantics per the Step 2A.4 rule -- never claiming exhaustive/complete
    look-through regardless of lookthrough_coverage."""
    single_name_flag = next((f for f in risk_flags if f.key == "single_name"), None)
    if single_name_flag is not None and true_exposures:
        top = true_exposures[0]
        if top.indirect > 0 and (top.true - top.direct) >= _MATERIAL_INDIRECT_DELTA:
            text = (
                f"当前风险主要来自头部持仓集中：{top.ticker} 直接持仓占比约 {top.direct:.1%}，"
                f"ETF 穿透后已识别实际暴露达 ≥{top.true:.1%}。"
            )
        else:
            text = f"当前风险主要来自头部持仓集中：{top.ticker} 直接持仓占比约 {top.direct:.1%}。"
        return {"label": "主要风险", "text": text}
    sector_flag = next((f for f in risk_flags if f.key == "sector"), None)
    if sector_flag is not None:
        return {"label": "主要风险", "text": f"当前风险主要来自板块集中度：{sector_flag.detail}"}
    # No single_name or sector flag -- including a liquidity-only flag set --
    # is a safe basis for concentration language. Fall through to the next
    # insight path rather than mislabeling liquidity risk as concentration.
    return None


def _diversification_insight(direct_effective_n: float | None) -> dict | None:
    """Priority D, used only when A does not apply. direct_effective_n is
    used strictly as DIRECT-holdings concentration evidence (Step 2A.4 scope
    rule) -- never translated into an overall/economic/factor diversification
    claim. None (no valid positions) fails closed rather than stating a
    meaningless "well diversified" claim about an empty portfolio."""
    if direct_effective_n is None:
        return None
    text = (
        f"直接持仓层面测算的有效持仓数量（Direct Effective N）约为 {direct_effective_n:.1f}，"
        "分布相对均衡，暂未发现明显的单一持仓集中问题。"
    )
    return {"label": "分散程度", "text": text}


def _asset_structure_insight(asset_allocation: dict[str, float]) -> dict | None:
    """Priority B (always attempted, second slot). Buckets the existing
    Equity/Fixed Income/Cash/Other weights (unchanged Layer 1 output) into
    one interpretive sentence -- current structure only, never a target or
    recommendation to maintain it (Step 2A.6 guardrail 1)."""
    equity = asset_allocation.get("Equity", 0.0)
    fixed_income = asset_allocation.get("Fixed Income", 0.0)
    cash = asset_allocation.get("Cash", 0.0)
    other = asset_allocation.get("Other", 0.0)
    non_equity = fixed_income + cash + other
    if equity <= 0 and non_equity <= 0:
        return None
    if equity >= 0.90 or non_equity < 0.05:
        base = "组合以股票资产为主，非股票资产占比较低"
    elif equity >= 0.50:
        base = "组合仍以股票资产为主，同时配置了一定比例的债券与现金类资产，整体结构并非纯股票组合"
    elif equity >= 0.35:
        base = "股票与债券类资产均有配置，组合结构相对均衡"
    else:
        base = "组合以债券与现金类资产为主，股票资产占比相对较低"
    cash_note = f"，另有约 {cash:.0%} 现金" if cash >= 0.05 else ""
    return {"label": "资产结构", "text": base + cash_note + "。"}


def build_portfolio_insights(result: AnalyticsResult) -> list[dict[str, str]]:
    """The single entry point the UI calls (Page 1, after Asset Allocation).
    Returns at most 2 items, each {"label": str, "text": str}; an empty list
    means the whole subsection should be omitted. Never fabricates a second
    item merely to reach 2 -- both slots can independently come back None."""
    primary = (
        _concentration_insight(result.risk_flags, result.true_exposures)
        or _diversification_insight(result.direct_effective_n)
    )
    structure = _asset_structure_insight(result.asset_allocation)
    insights = [item for item in (primary, structure) if item is not None]
    return insights[:2]
