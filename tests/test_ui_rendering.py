from __future__ import annotations

import re

import pytest

import ui


def test_rank_bars_preserves_order_and_values(mixed_result):
    rows = list(mixed_result.top_direct[:6])
    html = ui._rank_bars(rows)
    tickers_in_html = re.findall(r'<div class="rank-ticker">([^<]+)</div>', html)
    assert tickers_in_html == [ticker for ticker, _ in rows]
    pct_in_html = re.findall(r'<div class="rank-pct">([^<]+)</div>', html)
    assert pct_in_html == [f"{weight:.1%}" for _, weight in rows]


def test_rank_bars_fill_proportional_to_weight():
    rows = [("A", 0.30), ("B", 0.15), ("C", 0.03)]
    html = ui._rank_bars(rows)
    widths = [float(w) for w in re.findall(r'rank-fill" style="width:([\d.]+)%"', html)]
    assert widths[0] == pytest.approx(100.0)
    assert widths[1] == pytest.approx(50.0)
    assert widths[2] == pytest.approx(10.0)


def test_rank_bars_empty_is_empty_string():
    assert ui._rank_bars([]) == ""


def test_lookthrough_bars_preserves_order_and_true_total(mixed_result):
    exposures = [x for x in mixed_result.true_exposures if x.indirect > 0][:6]
    html = ui._lookthrough_bars(exposures)
    tickers_in_html = re.findall(r'<div class="etf-ticker">([^<]+)</div>', html)
    assert tickers_in_html == [x.ticker for x in exposures]
    totals_in_html = re.findall(r'<div class="etf-total">([^<]+)</div>', html)
    assert totals_in_html == [f"{x.true:.1%}" for x in exposures]


def test_lookthrough_bars_direct_and_indirect_segments_reflect_weights():
    from core.models import Exposure

    exposures = [Exposure("NVDA", 0.13, 0.026, 0.156)]
    html = ui._lookthrough_bars(exposures)
    direct_width = float(re.search(r'etf-direct" style="width:([\d.]+)%"', html).group(1))
    indirect_width = float(re.search(r'etf-indirect" style="width:([\d.]+)%"', html).group(1))
    assert direct_width == pytest.approx(0.13 / 0.156 * 100, abs=0.05)
    assert indirect_width == pytest.approx(0.026 / 0.156 * 100, abs=0.05)


def test_lookthrough_bars_empty_is_empty_string():
    assert ui._lookthrough_bars([]) == ""


def test_key_dates_html_empty_is_empty_string():
    """Step 2A.5: no relevant future events -> the whole subsection is
    omitted, never an empty heading."""
    assert ui._key_dates_html([]) == ""


def test_key_dates_html_renders_date_title_and_note():
    from datetime import date

    from core.key_dates import SecurityKeyDate

    events = [
        SecurityKeyDate(date(2026, 9, 16), "fomc", "美联储 FOMC 利率决议", "confirmed", "关注利率路径", "fomc_reference"),
        SecurityKeyDate(date(2026, 11, 17), "earnings", "NVDA 下一季度财报", "estimated", "关注集中度", "yfinance"),
    ]
    html = ui._key_dates_html(events)
    assert "关键关注日期" in html
    assert "2026-09-16" in html and "美联储 FOMC 利率决议" in html and "关注利率路径" in html
    assert "预计 2026-11-17" in html and "NVDA 下一季度财报" in html and "关注集中度" in html


def test_key_dates_html_preserves_order():
    from datetime import date

    from core.key_dates import SecurityKeyDate

    events = [
        SecurityKeyDate(date(2026, 9, 16), "fomc", "第一项", "confirmed", "n", "s"),
        SecurityKeyDate(date(2026, 11, 17), "earnings", "第二项", "estimated", "n", "s"),
    ]
    html = ui._key_dates_html(events)
    assert html.index("第一项") < html.index("第二项")


def test_portfolio_insights_html_empty_is_empty_string():
    """Step 2A.6: no supportable insight -> the whole block is omitted,
    never an empty heading."""
    assert ui._portfolio_insights_html([]) == ""


def test_portfolio_insights_html_renders_label_and_text():
    insights = [
        {"label": "主要风险", "text": "当前风险主要来自头部持仓集中：NVDA 直接持仓占比约 21.5%。"},
        {"label": "资产结构", "text": "组合仍以股票资产为主，同时配置了一定比例的债券与现金类资产。"},
    ]
    html = ui._portfolio_insights_html(insights)
    assert "组合洞察（PORTFOLIO INSIGHTS）" in html
    assert "主要风险" in html and "NVDA 直接持仓占比约 21.5%" in html
    assert "资产结构" in html and "并非纯股票组合" not in html  # sanity: text is passed through verbatim
    assert "债券与现金类资产" in html


def test_portfolio_insights_html_preserves_order():
    insights = [{"label": "A标签", "text": "第一条"}, {"label": "B标签", "text": "第二条"}]
    html = ui._portfolio_insights_html(insights)
    assert html.index("第一条") < html.index("第二条")


# --- Step 2A.8 Part B: compact Market Regime module -------------------------

class _FakeRegimeSnapshot:
    def __init__(self, status="complete", current_regime_label="偏多", correction="0-16%", bear="0-10%"):
        self.status = status
        self.current_regime_label = current_regime_label
        self.correction_risk_3m_display = correction
        self.bear_risk_6m_display = bear


def test_market_regime_html_omitted_when_unavailable():
    """Fail-closed: an unavailable snapshot must be cleanly omitted from
    Page 1, never rendered as a broken/empty block."""
    html = ui._market_regime_html(_FakeRegimeSnapshot(status="unavailable"))
    assert html == ""


def test_market_regime_html_renders_regime_and_risk_ranges():
    html = ui._market_regime_html(_FakeRegimeSnapshot())
    assert "市场状态（MARKET REGIME）" in html
    assert "偏多" in html
    assert "3个月调整风险" in html and "0-16%" in html
    assert "6个月熊市风险" in html and "0-10%" in html


def test_market_regime_html_never_contains_action_wording():
    """Market Regime describes market weather only -- the rendered block
    must never contain a buy/sell/reduce-type instruction."""
    html = ui._market_regime_html(_FakeRegimeSnapshot())
    for forbidden in ("买入", "卖出", "减仓", "加仓", "应该卖出股票"):
        assert forbidden not in html


def test_market_regime_html_uses_lighter_visual_class_than_kpi_metric_grid():
    """The module must stay visually lighter than the KPI row's large
    2rem .metric-value figures -- it reuses .regime-row/.regime-badge, not
    .metric-grid/.metric-value."""
    html = ui._market_regime_html(_FakeRegimeSnapshot())
    assert "metric-value" not in html
    assert "regime-row" in html and "regime-badge" in html

    for status, label in [("partial", "中性"), ("complete", "风险升高"), ("complete", "防御")]:
        assert label in ui._market_regime_html(_FakeRegimeSnapshot(status=status, current_regime_label=label))


# --- Step 2A.9: Analyst Consensus + 12-Month Target -------------------------

from core.analyst_view import AnalystConsensus, AnalystTarget, AnalystView  # noqa: E402


def _consensus(**overrides):
    defaults = dict(buy=57, hold=2, sell=1, total=60, label="强力买入")
    defaults.update(overrides)
    return AnalystConsensus(**defaults)


def _target(**overrides):
    defaults = dict(
        currency="USD", current=218.68, high=515.0, mean=327.6544, low=180.0,
        high_pct=135.5, mean_pct=49.8, low_pct=-17.7,
    )
    defaults.update(overrides)
    return AnalystTarget(**defaults)


def test_analyst_view_html_empty_when_view_none():
    assert ui._analyst_view_html(None) == ""


def test_analyst_view_html_empty_when_both_subsections_none():
    view = AnalystView("NVDA", consensus=None, target=None)
    assert ui._analyst_view_html(view) == ""


def test_analyst_view_html_renders_consensus_card():
    view = AnalystView("NVDA", consensus=_consensus(), target=None)
    html = ui._analyst_view_html(view)
    assert "分析师观点（ANALYST VIEW）" in html
    assert "分析师评级（ANALYST CONSENSUS）" in html
    assert "强力买入" in html
    assert "买入" in html and "57" in html
    assert "持有" in html and ">2<" in html
    assert "卖出" in html and ">1<" in html
    assert "评级覆盖：60 位分析师" in html


def test_analyst_view_html_omits_label_when_none():
    view = AnalystView("NVDA", consensus=_consensus(label=None), target=None)
    html = ui._analyst_view_html(view)
    assert "analyst-consensus-label" not in html
    assert "57" in html  # counts still render


def test_analyst_view_html_renders_target_card_with_pct():
    view = AnalystView("NVDA", consensus=None, target=_target())
    html = ui._analyst_view_html(view)
    assert "12个月目标价（12-MONTH TARGET）" in html
    assert "$515.00" in html and "+135.5%" in html
    assert "$327.65" in html and "+49.8%" in html
    assert "$180.00" in html and "-17.7%" in html
    assert "$218.68" in html
    assert "目标价基于最近可用数据" in html


def test_analyst_view_html_negative_pct_uses_negative_class():
    view = AnalystView("NVDA", consensus=None, target=_target())
    html = ui._analyst_view_html(view)
    assert "analyst-pct-neg" in html
    idx = html.index("-17.7%")
    assert "analyst-pct-neg" in html[max(0, idx - 60):idx]


def test_analyst_view_html_non_usd_currency_shown_natively():
    view = AnalystView("VDY", consensus=None, target=_target(currency="CAD"))
    html = ui._analyst_view_html(view)
    assert "CAD 515.00" in html
    assert "$515.00" not in html


def test_analyst_view_html_consensus_failure_does_not_block_target():
    view = AnalystView("NVDA", consensus=None, target=_target())
    html = ui._analyst_view_html(view)
    assert "分析师观点（ANALYST VIEW）" in html
    assert "12个月目标价" in html
    assert "ANALYST CONSENSUS" not in html


def test_analyst_view_html_target_failure_does_not_block_consensus():
    view = AnalystView("NVDA", consensus=_consensus(), target=None)
    html = ui._analyst_view_html(view)
    assert "分析师观点（ANALYST VIEW）" in html
    assert "分析师评级（ANALYST CONSENSUS）" in html
    assert "12-MONTH TARGET" not in html


def test_analyst_view_html_preserves_consensus_before_target_order():
    view = AnalystView("NVDA", consensus=_consensus(), target=_target())
    html = ui._analyst_view_html(view)
    assert html.index("ANALYST CONSENSUS") < html.index("12-MONTH TARGET")


# --- Step 2A.11: ETF Top Holdings -------------------------------------------

from core.etf_holdings import ETFHoldingRow, ETFTopHoldings  # noqa: E402


def _holdings_view(**overrides):
    rows = overrides.pop("holdings", (
        ETFHoldingRow("AAPL", "Apple Inc.", 0.07),
        ETFHoldingRow("MSFT", "Microsoft Corporation", 0.066),
    ))
    defaults = dict(
        ticker="VOO", holdings=rows,
        displayed_total=sum(r.weight for r in rows), total_reference_count=15,
    )
    defaults.update(overrides)
    return ETFTopHoldings(**defaults)


def test_etf_top_holdings_html_empty_when_view_none():
    assert ui._etf_top_holdings_html(None) == ""


def test_etf_top_holdings_html_empty_when_no_holdings():
    view = ETFTopHoldings(ticker="VOO", holdings=(), displayed_total=0.0, total_reference_count=0)
    assert ui._etf_top_holdings_html(view) == ""


def test_etf_top_holdings_html_renders_ticker_name_and_weight():
    html = ui._etf_top_holdings_html(_holdings_view())
    assert "主要持仓（Top Holdings）" in html
    assert "AAPL" in html and "Apple Inc." in html and "7.0%" in html
    assert "MSFT" in html and "Microsoft Corporation" in html and "6.6%" in html


def test_etf_top_holdings_html_omits_name_when_none():
    rows = (ETFHoldingRow("XYZ", None, 0.05),)
    html = ui._etf_top_holdings_html(_holdings_view(holdings=rows, total_reference_count=1))
    assert "XYZ" in html and "5.0%" in html
    # No stray "None" text where a missing name would otherwise render.
    assert "None" not in html


def test_etf_top_holdings_html_uses_top10_label_when_ten_or_more_reference_holdings():
    html = ui._etf_top_holdings_html(_holdings_view(total_reference_count=15))
    assert "前十大持仓合计" in html
    assert "已显示持仓合计" not in html


def test_etf_top_holdings_html_uses_partial_label_when_fewer_than_ten_reference_holdings():
    """Step 2A.11: never claim "Top 10" when the ETF's own reference
    snapshot has fewer than 10 constituents in total -- even though every
    one of them is displayed."""
    html = ui._etf_top_holdings_html(_holdings_view(total_reference_count=5))
    assert "已显示持仓合计" in html
    assert "前十大持仓合计" not in html


def test_etf_top_holdings_html_shows_reference_data_caption():
    html = ui._etf_top_holdings_html(_holdings_view())
    assert "基于当前可验证的参考持仓数据" in html


def test_etf_top_holdings_html_never_claims_full_completeness():
    """Never imply exhaustive/complete constituent coverage -- this is a
    frozen Top-N reference snapshot, not full ETF holdings data."""
    html = ui._etf_top_holdings_html(_holdings_view())
    for forbidden in ("完整持仓", "全部持仓", "完整前十大", "完整穿透", "Top 10 holdings"):
        assert forbidden not in html


def test_etf_top_holdings_html_total_calculated_correctly():
    rows = (ETFHoldingRow("AAPL", "Apple Inc.", 0.07), ETFHoldingRow("MSFT", "Microsoft Corporation", 0.066))
    html = ui._etf_top_holdings_html(_holdings_view(holdings=rows, displayed_total=0.136))
    assert "13.6%" in html
