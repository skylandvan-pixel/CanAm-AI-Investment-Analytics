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
