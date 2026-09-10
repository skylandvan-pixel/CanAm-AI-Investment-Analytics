"""Step 2A.6 -- Deterministic Portfolio Insights.

All fixtures are real core.analytics.analyze() outputs built from synthetic
holdings -- no invented AnalyticsResult fields, no LLM/network dependency.
"""

from __future__ import annotations

import pytest

from core.analytics import analyze, build_snapshot
from core.models import HoldingInput
from core.portfolio_insights import (
    _asset_structure_insight,
    _concentration_insight,
    _diversification_insight,
    build_portfolio_insights,
)


def _qf(quote_factory, *tickers):
    return quote_factory(*tickers)


# --- Overall shape: max 2, no filler ------------------------------------------

def test_max_two_insights(mixed_result):
    assert len(build_portfolio_insights(mixed_result)) <= 2


def test_no_filler_second_insight_when_structure_not_computable():
    """If asset_allocation is empty (degenerate), the structure slot must
    come back None rather than a fabricated line -- only the dynamic slot
    (if any) is shown."""
    assert _asset_structure_insight({}) is None


def test_empty_asset_allocation_yields_at_most_one_insight(mixed_result):
    import dataclasses

    degenerate = dataclasses.replace(mixed_result, asset_allocation={})
    insights = build_portfolio_insights(degenerate)
    assert len(insights) <= 1


# --- Asset-allocation insight --------------------------------------------------

def test_equity_dominant_with_fixed_income_structure_insight():
    insight = _asset_structure_insight({"Equity": 0.79, "Fixed Income": 0.21})
    assert insight["label"] == "资产结构"
    assert "并非纯股票组合" in insight["text"]


def test_near_pure_equity_structure_insight():
    insight = _asset_structure_insight({"Equity": 1.0})
    assert "非股票资产占比较低" in insight["text"]


def test_balanced_equity_fixed_income_structure_insight():
    insight = _asset_structure_insight({"Equity": 0.4, "Fixed Income": 0.6})
    assert "相对均衡" in insight["text"]


def test_fixed_income_dominant_structure_insight():
    insight = _asset_structure_insight({"Equity": 0.2, "Fixed Income": 0.8})
    assert "以债券与现金类资产为主" in insight["text"]


def test_material_cash_gets_its_own_mention():
    insight = _asset_structure_insight({"Equity": 0.9, "Cash": 0.10})
    assert "现金" in insight["text"]


# --- Concentration / risk insight ----------------------------------------------

def test_high_concentration_insight_uses_demo_portfolio(mixed_result):
    """mixed_result = NVDA(20)/VOO(50)/SGOV(30) -- NVDA true exposure
    (direct + VOO look-through) exceeds .15, triggering a high single_name
    flag."""
    insights = build_portfolio_insights(mixed_result)
    risk = next((i for i in insights if i["label"] == "主要风险"), None)
    assert risk is not None
    assert "NVDA" in risk["text"]
    assert "头部持仓集中" in risk["text"]


def test_low_concentration_portfolio_does_not_manufacture_warning(quote_factory):
    """A genuinely diversified, no-flag portfolio must get the
    diversification insight, never a fabricated concentration warning."""
    tickers = ["AAPL", "JPM", "XOM", "UNH", "LLY", "AGG", "BND", "SGOV"]
    holdings = [
        HoldingInput("AAPL", 6), HoldingInput("JPM", 6), HoldingInput("XOM", 6),
        HoldingInput("UNH", 6), HoldingInput("LLY", 6),
        HoldingInput("AGG", 15), HoldingInput("BND", 15), HoldingInput("SGOV", 20),
    ]
    result = analyze(build_snapshot(holdings, _qf(quote_factory, *tickers)))
    assert result.risk_flags == ()
    insights = build_portfolio_insights(result)
    assert not any(i["label"] == "主要风险" for i in insights)
    assert any(i["label"] == "分散程度" for i in insights)


def test_concentration_insight_none_when_no_risk_flags(mixed_result):
    assert _concentration_insight((), mixed_result.true_exposures) is None


# --- Risk-flag routing (Step 2A.6 Final Semantic Fix Pass) ----------------------
#
# single_name and sector flags may produce concentration language; a
# liquidity-only flag set must NEVER be worded as a concentration finding --
# it falls through to the next safe insight path instead.

def test_single_name_flag_produces_concentration_insight():
    from core.models import Exposure, RiskFlag

    flags = (RiskFlag("single_name", "单一标的集中度过高", "NVDA 是当前组合最主要的单一股票集中风险。", "high"),)
    exposures = (Exposure("NVDA", 0.20, 0.0, 0.20),)
    insight = _concentration_insight(flags, exposures)
    assert insight is not None
    assert insight["label"] == "主要风险"
    assert "头部持仓集中" in insight["text"]


def test_sector_flag_produces_concentration_insight_when_no_single_name():
    from core.models import RiskFlag

    flags = (RiskFlag("sector", "科技板块集中度过高", "直接持仓中科技板块占比为 45.0%。", "high"),)
    insight = _concentration_insight(flags, ())
    assert insight is not None
    assert insight["label"] == "主要风险"
    assert "板块集中度" in insight["text"]
    assert "45.0%" in insight["text"]


def test_liquidity_only_flag_does_not_produce_concentration_language():
    """A liquidity-only flag set (no single_name, no sector) must return
    None from the concentration path -- liquidity is a distinct risk
    dimension and must never be worded as concentration."""
    from core.models import RiskFlag

    flags = (RiskFlag("liquidity", "现金储备偏低", "现金及类现金储备占比为 2.0%。"),)
    assert _concentration_insight(flags, ()) is None


def test_liquidity_only_flag_falls_through_to_diversification_insight(quote_factory):
    """End-to-end: a portfolio whose only risk_flag is liquidity must get the
    diversification insight (D), never a fabricated concentration warning
    and never a liquidity-worded "主要风险" entry. 10 equal-weight stocks
    (10% each, exactly at -- not above -- the single_name threshold) spread
    across sectors (only 2 of 10 in Technology, under the 25% sector
    threshold) with no cash-like holdings -> liquidity is the only flag."""
    tickers = ["AAPL", "JPM", "XOM", "UNH", "LLY", "V", "AMZN", "NFLX", "MSFT", "BRK-B"]
    holdings = [HoldingInput(t, 10) for t in tickers]
    result = analyze(build_snapshot(holdings, _qf(quote_factory, *tickers)))
    assert [f.key for f in result.risk_flags] == ["liquidity"]
    insights = build_portfolio_insights(result)
    assert not any(i["label"] == "主要风险" for i in insights)
    assert any(i["label"] == "分散程度" for i in insights)


def test_single_name_takes_priority_over_coexisting_sector_flag():
    """core.demo's real demo portfolio (VOO/XLK/NVDA/SGOV/AAPL) carries both
    single_name (NVDA) and sector (Technology, via XLK+NVDA) flags --
    single_name must win the concentration slot."""
    from core.demo import DEMO_HOLDINGS, DEMO_QUOTES

    result = analyze(build_snapshot(DEMO_HOLDINGS, DEMO_QUOTES))
    assert {f.key for f in result.risk_flags} >= {"single_name", "sector"}
    insight = _concentration_insight(result.risk_flags, result.true_exposures)
    assert "NVDA" in insight["text"]
    assert "头部持仓集中" in insight["text"]


# --- ETF overlap / true-exposure (lower-bound semantics) -----------------------

def test_etf_overlap_insight_when_indirect_exposure_is_material(mixed_result):
    """Step 2A.4 regression: the look-through clause must appear when
    indirect exposure materially raises true exposure above direct, and it
    must preserve "≥" lower-bound semantics -- never an exact-equals claim."""
    top = mixed_result.true_exposures[0]
    assert top.ticker == "NVDA" and top.indirect > 0.01
    insight = _concentration_insight(mixed_result.risk_flags, mixed_result.true_exposures)
    assert "≥" in insight["text"]
    assert "ETF 穿透后" in insight["text"]


def test_no_lookthrough_clause_when_indirect_is_immaterial(quote_factory):
    """A single_name flag with no (or negligible) ETF-derived contribution
    must not mention look-through at all -- nothing to add."""
    from core.models import Exposure, RiskFlag

    flags = (RiskFlag("single_name", "单一标的集中度过高", "NVDA 是当前组合最主要的单一股票集中风险。", "high"),)
    exposures = (Exposure("NVDA", 0.20, 0.0, 0.20),)
    insight = _concentration_insight(flags, exposures)
    assert "≥" not in insight["text"]
    assert "ETF 穿透" not in insight["text"]


def test_partial_lookthrough_preserves_qualified_language(quote_factory):
    """Even when lookthrough_coverage is "insufficient", the concentration
    insight must still use "≥" (never drop the qualifier merely because
    coverage happens to be incomplete -- Step 2A.4 principle preserved)."""
    holdings = [HoldingInput("NVDA", 10), HoldingInput("XLF", 40), HoldingInput("CBIL", 30)]
    result = analyze(build_snapshot(holdings, _qf(quote_factory, "NVDA", "XLF", "CBIL")))
    insight = _concentration_insight(result.risk_flags, result.true_exposures)
    if insight is not None and result.true_exposures and result.true_exposures[0].indirect > 0:
        assert "≥" in insight["text"]


def test_no_false_exhaustive_lookthrough_language(mixed_result):
    """No wording anywhere in the concentration insight may claim exhaustive/
    complete look-through coverage."""
    insight = _concentration_insight(mixed_result.risk_flags, mixed_result.true_exposures)
    for forbidden in ("完全穿透", "全部真实暴露", "完整穿透", "真实暴露就是"):
        assert forbidden not in insight["text"]


# --- Direct Effective N scope ---------------------------------------------------

def test_direct_effective_n_scope_not_overstated():
    """The diversification insight must scope Direct Effective N to direct
    holdings explicitly and must never claim overall/economic/factor
    diversification."""
    insight = _diversification_insight(5.76)
    assert "直接持仓层面" in insight["text"]
    assert "Direct Effective N" in insight["text"]
    for forbidden in ("整体分散", "经济分散", "因子分散", "真实分散"):
        assert forbidden not in insight["text"]


def test_diversification_insight_none_without_positions():
    assert _diversification_insight(None) is None


# --- Current fact vs target / recommendation leakage ---------------------------

def test_asset_structure_never_recommends_maintaining_allocation():
    insight = _asset_structure_insight({"Equity": 0.79, "Fixed Income": 0.21})
    for forbidden in ("应维持", "建议保持", "配置合理", "良好的防守基石", "适合作为目标配置"):
        assert forbidden not in insight["text"]


def test_no_recommendation_language_anywhere(mixed_result):
    insights = build_portfolio_insights(mixed_result)
    forbidden_terms = ("应该", "建议", "需要减仓", "需要增持", "维持当前配置", "调整到", "目标比例", "最优配置")
    for item in insights:
        for term in forbidden_terms:
            assert term not in item["text"], f"{term!r} leaked into {item}"


# --- No currency-exposure branch (Step 2A.6 Final Semantic Fix Pass) -----------
#
# Position.currency is the resolved quote/settlement currency, not a
# canonical economic-currency-exposure field -- a CAD-listed, USD-index-
# tracking ETF like ZSP would be wrongly counted as "CAD exposure" under a
# settlement-currency rule. No such economic-exposure field exists on
# AnalyticsResult, so this module must have NO currency-insight branch at
# all, regardless of how currency-concentrated a portfolio's settlement
# currencies are.

def test_no_currency_insight_function_exists():
    """The removed _currency_insight helper must not be importable -- the
    branch is gone entirely, not merely disabled."""
    import core.portfolio_insights as module

    assert not hasattr(module, "_currency_insight")


def test_quote_currency_never_produces_a_currency_label(mixed_result):
    """mixed_result is 100% USD-settled; build_portfolio_insights must never
    emit a currency-exposure-labeled insight from settlement currency."""
    insights = build_portfolio_insights(mixed_result)
    assert not any(i["label"] == "货币暴露" for i in insights)


def test_cad_listed_us_index_etf_creates_no_false_currency_insight(quote_factory):
    """ZSP is CAD-listed/settled but tracks the S&P 500 (genuinely USD
    economic exposure) -- a settlement-currency rule would wrongly treat
    this all-ZSP, CAD-account portfolio as "100% CAD exposure" (or, read the
    other way, would need to know ZSP's true index currency to get it right
    at all). Because no currency-insight branch exists, this discrepancy
    can never surface as a wrong claim in either direction."""
    holdings = [HoldingInput("ZSP", 50, currency="CAD")]
    result = analyze(build_snapshot(holdings, _qf(quote_factory, "ZSP"), account_currency="CAD"))
    insights = build_portfolio_insights(result)
    assert not any(i["label"] == "货币暴露" for i in insights)
    for item in insights:
        assert "经济暴露" not in item["text"]
        assert "USD/CAD" not in item["text"] and "CAD/USD" not in item["text"]


def test_high_settlement_currency_concentration_still_produces_no_currency_insight(quote_factory):
    """Even a portfolio whose settlement currency is overwhelmingly non-
    account-currency (the exact scenario the old, removed branch would have
    flagged) must not produce a 货币暴露 insight -- there is no safe
    inference from settlement currency alone."""
    tickers = ["AAPL", "MSFT", "JPM", "V", "BRK-B", "XOM", "UNH", "LLY", "AMZN", "NFLX", "TSLA"]
    holdings = [HoldingInput(t, 9) for t in tickers]
    result = analyze(build_snapshot(holdings, _qf(quote_factory, *tickers), cash=750, account_currency="CAD", usd_cad=1.35))
    insights = build_portfolio_insights(result)
    assert not any(i["label"] == "货币暴露" for i in insights)


# --- Empty-safe behavior ---------------------------------------------------------

def test_build_portfolio_insights_is_empty_safe_for_degenerate_result(mixed_result):
    import dataclasses

    degenerate = dataclasses.replace(
        mixed_result, asset_allocation={}, risk_flags=(), true_exposures=(), direct_effective_n=None,
    )
    assert build_portfolio_insights(degenerate) == []


# --- Schema safety regression -----------------------------------------------------

def test_committee_result_schema_unchanged_by_portfolio_insights():
    """Step 2A.6 is a Page 1 / Layer 1 feature only -- it must never touch
    the Gemini-facing CommitteeResult schema (core/ai.py is untouched by
    this module entirely). Byte count rebaselined to 4072 in Step 2A.10.1
    (CommitteeMember.stance dropped "维持配置" -- a deliberate, approved,
    unrelated schema change; see tests/test_ai.py::
    test_schema_bytes_match_step_2a_3_approved_baseline)."""
    import json

    from core.ai import CommitteeResult

    schema = CommitteeResult.model_json_schema()
    assert len(json.dumps(schema, ensure_ascii=False).encode("utf-8")) == 4072
    assert len(schema.get("$defs", {})) == 4
