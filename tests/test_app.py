from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app():
    return AppTest.from_file(APP_PATH).run(timeout=20)


def test_app_startup_smoke():
    app = _app()
    assert not app.exception
    assert any("加美通 AI 投资分析" in item.value for item in app.markdown)


def test_page1_smoke_shows_chinese_kpis():
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    assert "投资组合评分" in joined
    assert "投资组合市值" in joined
    assert "风险等级" in joined
    assert "前三大持仓占比" in joined


def test_page1_old_low_value_summary_sentence_is_gone():
    """Step 2A.6: the old "组合以股票（Equity）为主，前三大直接持仓合计 XX%。"
    sentence must no longer render anywhere on Page 1."""
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    assert "组合以" not in joined
    assert "直接持仓合计" not in joined


def test_page1_shows_portfolio_insights_block():
    """Step 2A.6: the new 组合洞察（PORTFOLIO INSIGHTS）block replaces the old
    summary sentence, appearing after Asset Allocation, with at most 2 short
    insights built entirely from deterministic Layer 1 data (the demo
    portfolio's real NVDA concentration + asset-allocation structure)."""
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    assert "组合洞察（PORTFOLIO INSIGHTS）" in joined
    assert "主要风险" in joined and "NVDA" in joined
    assert "资产结构" in joined


def test_page1_portfolio_insights_uses_no_recommendation_language():
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    idx = joined.index("组合洞察（PORTFOLIO INSIGHTS）")
    block = joined[idx:idx + 400]
    for forbidden in ("应该", "建议", "维持当前配置", "目标比例", "最优配置"):
        assert forbidden not in block


def _fake_regime_snapshot(status="complete", label="偏多", correction="0-16%", bear="0-10%"):
    from core.market_regime import MarketRegimeSnapshot

    return MarketRegimeSnapshot(
        status=status, current_regime="Bull", current_regime_label=label,
        correction_risk_3m_display=correction, bear_risk_6m_display=bear,
    )


def test_page1_market_regime_appears_between_kpi_and_treemap_header(monkeypatch):
    """Step 2A.8 placement requirement: KPI cards -> Market Regime ->
    Direct Holdings Treemap, in that order."""
    import ui

    monkeypatch.setattr(ui, "get_market_regime_snapshot", lambda *a, **k: _fake_regime_snapshot())
    app = _app()
    assert not app.exception
    values = [item.value for item in app.markdown]
    kpi_idx = next(i for i, v in enumerate(values) if "metric-grid" in v)
    regime_idx = next(i for i, v in enumerate(values) if "市场状态（MARKET REGIME）" in v)
    treemap_idx = next(i for i, v in enumerate(values) if "直接持仓（Direct Holdings）" in v)
    assert kpi_idx < regime_idx < treemap_idx


def test_page1_market_regime_shows_regime_and_risk_ranges(monkeypatch):
    import ui

    monkeypatch.setattr(ui, "get_market_regime_snapshot", lambda *a, **k: _fake_regime_snapshot())
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    assert "市场状态（MARKET REGIME）" in joined
    assert "偏多" in joined
    assert "3个月调整风险" in joined and "0-16%" in joined
    assert "6个月熊市风险" in joined and "0-10%" in joined


def test_page1_renders_normally_without_market_regime_when_unavailable(monkeypatch):
    """Fail-closed: Market Regime failure must never break Page 1 -- the
    module is simply omitted, everything else renders unchanged."""
    import ui

    monkeypatch.setattr(
        ui, "get_market_regime_snapshot",
        lambda *a, **k: _fake_regime_snapshot(status="unavailable"),
    )
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    assert "市场状态（MARKET REGIME）" not in joined
    assert "投资组合评分" in joined
    assert "直接持仓（Direct Holdings）" in joined


def test_page1_market_regime_requires_no_ai_provider_call(monkeypatch):
    """Market Regime is Layer 1 / deterministic -- rendering it must never
    touch core.ai (no Gemini/Anthropic call)."""
    import core.ai
    import ui

    def _forbidden(*args, **kwargs):
        raise AssertionError("Page 1 rendering must never call run_ai_analysis")

    monkeypatch.setattr(core.ai, "run_ai_analysis", _forbidden)
    monkeypatch.setattr(ui, "get_market_regime_snapshot", lambda *a, **k: _fake_regime_snapshot())
    app = _app()
    assert not app.exception


def test_page1_market_regime_never_shows_buy_sell_recommendation(monkeypatch):
    import ui

    monkeypatch.setattr(ui, "get_market_regime_snapshot", lambda *a, **k: _fake_regime_snapshot(label="风险升高"))
    app = _app()
    assert not app.exception
    joined = "\n".join(item.value for item in app.markdown)
    idx = joined.index("市场状态（MARKET REGIME）")
    block = joined[idx:idx + 400]
    for forbidden in ("应该卖出股票", "买入", "卖出", "减仓", "加仓"):
        assert forbidden not in block


def test_portfolio_insights_selection_requires_no_gemini_or_anthropic_call(monkeypatch):
    """Step 2A.6: Portfolio Insights is Layer 1 / core product behavior --
    rendering Page 1 must never touch core.ai."""
    import core.ai

    def _forbidden(*args, **kwargs):
        raise AssertionError("Page 1 rendering must never call run_ai_analysis")

    monkeypatch.setattr(core.ai, "run_ai_analysis", _forbidden)
    app = _app()
    assert not app.exception


def test_only_one_analyze_button_exists():
    """P0 regression: there must be exactly one '开始分析' button anywhere
    in the sidebar, not one inside the import panel and another below it."""
    app = _app()
    assert not app.exception
    matches = [b for b in app.button if b.label == "开始分析"]
    assert len(matches) == 1


def test_only_one_holdings_table_exists():
    """P0 regression: import must not create a second, separately-rendered
    preview table alongside the editable holdings table."""
    app = _app()
    assert not app.exception
    holdings_tables = [d for d in app.dataframe if d.key == "portfolio_editor"]
    assert len(holdings_tables) == 1
    other_tables = [d for d in app.dataframe if d.key != "portfolio_editor"]
    assert len(other_tables) == 0


def test_holdings_table_has_no_per_row_currency_or_account_type_columns():
    app = _app()
    assert not app.exception
    editor = next(d for d in app.dataframe if d.key == "portfolio_editor")
    assert list(editor.value.columns) == ["Ticker", "Quantity", "Average Cost"]


def test_account_type_is_a_single_portfolio_level_selectbox():
    app = _app()
    assert not app.exception
    matches = [s for s in app.selectbox if "账户类型" in s.label]
    assert len(matches) == 1
    assert matches[0].options == ["TFSA（免税账户）", "Taxable（应税账户）"]
    assert matches[0].value == "Taxable"


def test_csv_import_populates_canonical_table_directly():
    """IMPORT -> UI STATE: the exact same table the user edits must show
    the imported tickers/quantities immediately, no separate preview."""
    app = _app()
    csv = b"Ticker,Quantity,Average Cost\nNVDA,10,100\nVOO,20,400\nSGOV,30,100\n"
    app.file_uploader[0].upload("holdings.csv", csv, "text/csv").run(timeout=20)
    assert not app.exception
    editor = next(d for d in app.dataframe if d.key == "portfolio_editor")
    frame = editor.value
    assert list(frame["Ticker"]) == ["NVDA", "VOO", "SGOV"]
    assert list(frame["Quantity"]) == [10, 20, 30]
    assert "FINN" not in set(frame["Ticker"])
    assert "SPMO" not in set(frame["Ticker"])
    # Only one Analyze button must exist even with the import panel open and populated.
    assert len([b for b in app.button if b.label == "开始分析"]) == 1


def test_import_replaces_demo_portfolio_entirely():
    app = _app()
    demo_editor = next(d for d in app.dataframe if d.key == "portfolio_editor")
    assert "AAPL" in set(demo_editor.value["Ticker"])  # confirm demo was showing first
    csv = b"Ticker,Quantity\nNVDA,10\nVOO,20\nSGOV,30\n"
    app.file_uploader[0].upload("holdings.csv", csv, "text/csv").run(timeout=20)
    assert not app.exception
    editor = next(d for d in app.dataframe if d.key == "portfolio_editor")
    tickers = set(editor.value["Ticker"])
    assert tickers == {"NVDA", "VOO", "SGOV"}
    assert "AAPL" not in tickers and "XLK" not in tickers


def test_rerun_after_import_does_not_restore_demo_state():
    """A page-switch or any other rerun must not silently revert an
    imported portfolio back to the demo holdings."""
    app = _app()
    csv = b"Ticker,Quantity\nNVDA,10\nVOO,20\nSGOV,30\n"
    app.file_uploader[0].upload("holdings.csv", csv, "text/csv").run(timeout=20)
    app.run(timeout=20)  # a second, unrelated rerun
    assert not app.exception
    editor = next(d for d in app.dataframe if d.key == "portfolio_editor")
    assert set(editor.value["Ticker"]) == {"NVDA", "VOO", "SGOV"}


def test_switching_to_a_different_csv_replaces_rather_than_merges():
    app = _app()
    csv_a = b"Ticker,Quantity\nNVDA,10\nVOO,20\n"
    app.file_uploader[0].upload("holdings_a.csv", csv_a, "text/csv").run(timeout=20)
    editor_a = next(d for d in app.dataframe if d.key == "portfolio_editor")
    assert set(editor_a.value["Ticker"]) == {"NVDA", "VOO"}

    csv_b = b"Ticker,Quantity\nAAPL,5\nSGOV,15\n"
    app.file_uploader[0].upload("holdings_b.csv", csv_b, "text/csv").run(timeout=20)
    assert not app.exception
    editor_b = next(d for d in app.dataframe if d.key == "portfolio_editor")
    tickers_b = set(editor_b.value["Ticker"])
    assert tickers_b == {"AAPL", "SGOV"}
    assert "NVDA" not in tickers_b and "VOO" not in tickers_b


def test_imported_holdings_pending_analysis_does_not_crash_or_use_stale_quotes():
    """Right after import, quotes for the new tickers have not been fetched
    yet (that only happens on Analyze click). The app must show a plain
    prompt, never an error and never a result computed from old quotes."""
    app = _app()
    csv = b"Ticker,Quantity\nNVDA,10\nVOO,20\nSGOV,30\n"
    app.file_uploader[0].upload("holdings.csv", csv, "text/csv").run(timeout=20)
    assert not app.exception
    assert any("开始分析" in i.value for i in app.info)


def test_csv_import_with_unrecognized_headers_shows_minimal_manual_mapping():
    app = _app()
    csv = b"Name,Value\nVOO,10\nAAPL,5\n"
    app.file_uploader[0].upload("holdings.csv", csv, "text/csv").run(timeout=20)
    assert not app.exception
    assert any("请手动选择" in c.value for c in app.caption)
    assert {"代码列（Ticker）", "数量列（Quantity）"} <= {s.label for s in app.selectbox}
    # Still just the one canonical table, showing demo data until mapping is resolved.
    assert len([d for d in app.dataframe if d.key == "portfolio_editor"]) == 1


def test_xlsx_import_populates_canonical_table(tmp_path):
    path = tmp_path / "holdings.xlsx"
    pd.DataFrame([
        {"Ticker": "NVDA", "Quantity": 10, "Average Cost": 100},
        {"Ticker": "VOO", "Quantity": 20, "Average Cost": 400},
        {"Ticker": "SGOV", "Quantity": 30, "Average Cost": 100},
    ]).to_excel(path, index=False)
    app = _app()
    app.file_uploader[0].upload(
        "holdings.xlsx", path.read_bytes(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ).run(timeout=20)
    assert not app.exception
    editor = next(d for d in app.dataframe if d.key == "portfolio_editor")
    frame = editor.value
    assert list(frame["Ticker"]) == ["NVDA", "VOO", "SGOV"]
    assert list(frame["Quantity"]) == [10, 20, 30]


def test_import_to_analytics_chain_matches_exactly_for_csv():
    """The regression the P0 report called out: trace CSV bytes all the way
    through the same functions app.py uses (read -> map -> build_preview ->
    _holdings_from_frame -> build_snapshot -> analyze) with fixed fake quotes
    standing in for the network fetch, and assert the analyzed tickers are
    exactly the imported ones -- no FINN/SPMO/AAPL/demo leftovers."""
    import sys
    sys.path.insert(0, str(APP_PATH.parent))
    from app import _holdings_from_frame
    from core.analytics import analyze, build_snapshot
    from core.importers import auto_map_columns, build_preview, read_portfolio_file
    from core.models import Quote

    csv = b"Ticker,Quantity,Average Cost\nNVDA,10,100\nVOO,20,400\nSGOV,30,100\n"
    raw = read_portfolio_file(csv, "holdings.csv")
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)

    assert list(preview["Ticker"]) == ["NVDA", "VOO", "SGOV"]
    assert list(preview["Quantity"]) == [10, 20, 30]

    holdings = _holdings_from_frame(preview)
    assert [h.ticker for h in holdings] == ["NVDA", "VOO", "SGOV"]
    assert [h.quantity for h in holdings] == [10, 20, 30]

    quotes = {t: Quote(t, 100.0, "USD", 99.0, "test", "verified") for t in ("NVDA", "VOO", "SGOV")}
    result = analyze(build_snapshot(holdings, quotes))
    tickers = {t for t, _ in result.top_direct}
    assert tickers == {"NVDA", "VOO", "SGOV"}
    assert "FINN" not in tickers and "SPMO" not in tickers and "AAPL" not in tickers


def test_import_to_analytics_chain_matches_exactly_for_xlsx(tmp_path):
    import sys
    sys.path.insert(0, str(APP_PATH.parent))
    from app import _holdings_from_frame
    from core.analytics import analyze, build_snapshot
    from core.importers import auto_map_columns, build_preview, read_portfolio_file
    from core.models import Quote

    path = tmp_path / "holdings.xlsx"
    pd.DataFrame([
        {"Ticker": "NVDA", "Quantity": 10, "Average Cost": 100},
        {"Ticker": "VOO", "Quantity": 20, "Average Cost": 400},
        {"Ticker": "SGOV", "Quantity": 30, "Average Cost": 100},
    ]).to_excel(path, index=False)
    raw = read_portfolio_file(path.read_bytes(), "holdings.xlsx")
    mapping = auto_map_columns(raw)
    preview = build_preview(raw, mapping)
    holdings = _holdings_from_frame(preview)
    assert [h.ticker for h in holdings] == ["NVDA", "VOO", "SGOV"]

    quotes = {t: Quote(t, 100.0, "USD", 99.0, "test", "verified") for t in ("NVDA", "VOO", "SGOV")}
    result = analyze(build_snapshot(holdings, quotes))
    tickers = {t for t, _ in result.top_direct}
    assert tickers == {"NVDA", "VOO", "SGOV"}


def test_unresolved_ticker_identity_blocks_analysis_with_compact_error():
    """P0 regression: a ticker whose quote is a fail-closed identity
    mismatch (e.g. FINN colliding with an unrelated U.S. OTC stock) must
    show the compact error and never render the results pages with a
    fabricated valuation."""
    import pandas as pd
    from core.models import Quote

    app = AppTest.from_file(APP_PATH)
    frame = pd.DataFrame(
        [{"Ticker": "CBIL", "Quantity": 3218, "Average Cost": None},
         {"Ticker": "FINN", "Quantity": 3500, "Average Cost": None}],
        columns=["Ticker", "Quantity", "Average Cost"],
    )
    app.session_state["holdings"] = frame
    app.session_state["quotes"] = {
        "CBIL": Quote("CBIL", 50.01, "CAD", 49.5, "test", "verified"),
        "FINN": Quote("FINN", None, None, None, status="identity:FINN 无法可靠识别证券身份，请确认代码或交易市场。"),
    }
    app.session_state["analyzed_tickers"] = {"CBIL", "FINN"}
    app.session_state["is_demo"] = False
    app.session_state["cash"] = 0.0
    app.session_state["account_currency"] = "CAD"
    app.session_state["account_type"] = "Taxable"
    app.session_state["usd_cad"] = 1.36
    app.session_state["ai_unlocked"] = False
    app.session_state["ai_result"] = None
    app.run(timeout=20)
    assert not app.exception
    assert any("FINN 无法可靠识别证券身份，请确认代码或交易市场。" in e.value for e in app.error)
    joined = "\n".join(m.value for m in app.markdown)
    assert "投资组合评分" not in joined  # results pages must not render


def test_treemap_click_selection_shows_profile_directly_no_selectbox():
    """AppTest cannot simulate an actual Plotly click event, so this drives
    the same session_state the click handler writes to (treemap_selected)
    and asserts: (a) the old two-step selectbox is gone, (b) a selection
    renders the Security Profile immediately below the treemap with no
    further interaction needed."""
    app = _app()
    assert not app.exception
    assert not any(s.key == "treemap_ticker_select" for s in app.selectbox)
    joined = "\n".join(m.value for m in app.markdown)
    assert "选择持仓查看证券简介" not in joined

    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "英伟达" in joined and "NVIDIA" in joined
    assert "暂无详细证券介绍" not in joined


def test_treemap_no_selection_shows_no_profile_card():
    app = _app()
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "证券简介" not in joined


def test_treemap_selection_resets_when_portfolio_changes():
    """A profile shown for a ticker that no longer exists after a new
    import would be stale/misleading -- must be cleared."""
    app = _app()
    app.session_state["treemap_selected"] = "AAPL"
    app.run(timeout=20)
    csv = b"Ticker,Quantity\nNVDA,10\nVOO,20\nSGOV,30\n"
    app.file_uploader[0].upload("holdings.csv", csv, "text/csv").run(timeout=20)
    assert not app.exception
    assert app.session_state["treemap_selected"] is None


def test_security_profile_shows_key_dates_section_for_known_security(monkeypatch):
    """Step 2A.5: selecting a security whose deterministic Key Dates lookup
    returns relevant events shows the 关键关注日期（Key Dates）subsection
    inside the same Security Profile card, below the existing description/
    weight content."""
    import ui
    from datetime import date

    from core.key_dates import SecurityKeyDate

    monkeypatch.setattr(ui, "get_security_key_dates", lambda ticker: [
        SecurityKeyDate(date(2026, 9, 16), "fomc", "美联储 FOMC 利率决议", "confirmed", "关注利率路径", "fomc_reference"),
        SecurityKeyDate(date(2026, 11, 17), "earnings", "NVDA 下一季度财报", "estimated", "关注集中度变化", "yfinance"),
    ])
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "英伟达" in joined  # existing profile content still present
    assert "关键关注日期" in joined
    assert "美联储 FOMC 利率决议" in joined and "2026-09-16" in joined
    assert "NVDA 下一季度财报" in joined and "预计 2026-11-17" in joined


def test_key_dates_content_updates_when_selected_ticker_changes(monkeypatch):
    """Step 2A.5 requirement #15: the Key Dates list must update
    automatically when the selected security changes -- never show a
    previous ticker's stale dates."""
    import ui
    from datetime import date

    from core.key_dates import SecurityKeyDate

    def _fake(ticker):
        if ticker == "NVDA":
            return [SecurityKeyDate(date(2026, 11, 17), "earnings", "NVDA 下一季度财报", "estimated", "n", "yfinance")]
        if ticker == "AAPL":
            return [SecurityKeyDate(date(2026, 10, 29), "earnings", "AAPL 下一季度财报", "estimated", "n", "yfinance")]
        return []

    monkeypatch.setattr(ui, "get_security_key_dates", _fake)
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    joined_nvda = "\n".join(m.value for m in app.markdown)
    assert "NVDA 下一季度财报" in joined_nvda and "AAPL 下一季度财报" not in joined_nvda

    app.session_state["treemap_selected"] = "AAPL"
    app.run(timeout=20)
    joined_aapl = "\n".join(m.value for m in app.markdown)
    assert "AAPL 下一季度财报" in joined_aapl and "NVDA 下一季度财报" not in joined_aapl


def test_no_relevant_key_dates_omits_subsection_cleanly(monkeypatch):
    """Preferred fail-closed behavior: when there are no relevant future
    events, the Key Dates subsection is omitted entirely -- no "N/A",
    "Unknown", or "暂无数据"-style filler line."""
    import ui

    monkeypatch.setattr(ui, "get_security_key_dates", lambda ticker: [])
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "英伟达" in joined  # profile card itself still renders
    assert "关键关注日期" not in joined
    for forbidden in ("N/A", "Unknown", "数据不可用", "暂无关键日期"):
        assert forbidden not in joined


def test_key_dates_selection_requires_no_gemini_or_anthropic_call(monkeypatch):
    """Step 2A.5: Security Key Dates is Layer 1 / core product behavior --
    selecting a security on Page 1 must never touch core.ai, regardless of
    Pro unlock, beta code, or Gemini/Anthropic configuration."""
    import core.ai

    def _forbidden(*args, **kwargs):
        raise AssertionError("Page 1 security selection must never call run_ai_analysis")

    monkeypatch.setattr(core.ai, "run_ai_analysis", _forbidden)
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception


# --- Step 2A.9: Analyst Consensus + 12-Month Target -------------------------

def test_analyst_view_appears_for_eligible_stock_with_data(monkeypatch):
    """Step 2A.9: selecting an eligible stock with valid analyst data shows
    the 分析师观点（ANALYST VIEW）section below Key Dates, inside Page 1's
    existing Security Profile flow."""
    import ui
    from core.analyst_view import AnalystConsensus, AnalystTarget, AnalystView

    view = AnalystView(
        "NVDA",
        consensus=AnalystConsensus(buy=57, hold=2, sell=1, total=60, label="强力买入"),
        target=AnalystTarget(
            currency="USD", current=218.68, high=515.0, mean=327.6544, low=180.0,
            high_pct=135.5, mean_pct=49.8, low_pct=-17.7,
        ),
    )
    monkeypatch.setattr(ui, "get_analyst_view", lambda ticker: view)
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "分析师观点（ANALYST VIEW）" in joined
    assert "强力买入" in joined


def test_analyst_view_omitted_for_etf_with_no_live_network_call(monkeypatch):
    """Step 2A.9: an ETF is never eligible -- the real core function must
    return None before any yfinance network call is attempted, and Page 1
    must still render cleanly with the section simply omitted."""
    import ui
    import yfinance as yf
    from core.analyst_view import get_analyst_view as real_get_analyst_view

    def _forbidden_ticker(*args, **kwargs):
        raise AssertionError("must not construct yf.Ticker for an ineligible ETF")

    monkeypatch.setattr(yf, "Ticker", _forbidden_ticker)
    monkeypatch.setattr(ui, "get_analyst_view", real_get_analyst_view)
    app = _app()
    app.session_state["treemap_selected"] = "VOO"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "分析师观点" not in joined


def test_analyst_view_failure_does_not_break_security_profile(monkeypatch):
    """Fail-closed: when analyst data is unavailable, the rest of the
    Security Profile card (description, Key Dates) must still render."""
    import ui

    monkeypatch.setattr(ui, "get_analyst_view", lambda ticker: None)
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "英伟达" in joined and "NVIDIA" in joined
    assert "分析师观点" not in joined


def test_analyst_view_selection_requires_no_gemini_or_anthropic_call(monkeypatch):
    """Step 2A.9: Analyst View is Layer 1 market-consensus data -- selecting
    a security with analyst data present must never touch core.ai."""
    import core.ai
    import ui
    from core.analyst_view import AnalystConsensus, AnalystView

    def _forbidden(*args, **kwargs):
        raise AssertionError("Analyst View must never call run_ai_analysis")

    monkeypatch.setattr(core.ai, "run_ai_analysis", _forbidden)
    monkeypatch.setattr(ui, "get_analyst_view", lambda ticker: AnalystView(
        ticker, consensus=AnalystConsensus(buy=1, hold=1, sell=1, total=3, label="持有"), target=None,
    ))
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    assert not app.exception


def test_analyst_view_content_updates_when_selected_ticker_changes(monkeypatch):
    """The Analyst View content must update when the selected security
    changes -- never show a previous ticker's stale data."""
    import ui
    from core.analyst_view import AnalystConsensus, AnalystView

    def _fake(ticker):
        if ticker == "NVDA":
            return AnalystView(ticker, consensus=AnalystConsensus(buy=1, hold=0, sell=0, total=1, label="强力买入"), target=None)
        if ticker == "AAPL":
            return AnalystView(ticker, consensus=AnalystConsensus(buy=0, hold=1, sell=0, total=1, label="持有"), target=None)
        return None

    monkeypatch.setattr(ui, "get_analyst_view", _fake)
    app = _app()
    app.session_state["treemap_selected"] = "NVDA"
    app.run(timeout=20)
    joined_nvda = "\n".join(m.value for m in app.markdown)
    assert "强力买入" in joined_nvda

    app.session_state["treemap_selected"] = "AAPL"
    app.run(timeout=20)
    joined_aapl = "\n".join(m.value for m in app.markdown)
    assert "强力买入" not in joined_aapl


# --- Step 2A.11: ETF Top Holdings -------------------------------------------

def test_etf_top_holdings_renders_for_eligible_etf_with_reference_data():
    """VOO is a real demo-portfolio holding with 15 reference constituents
    (see core.reference.ETF_HOLDINGS) -- selecting it must show the Top
    Holdings subsection with real constituent tickers/weights, placed
    inside the existing Security Profile card."""
    app = _app()
    app.session_state["treemap_selected"] = "VOO"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "证券简介（Security Profile）" in joined
    assert "主要持仓（Top Holdings）" in joined
    assert "AAPL" in joined and "7.0%" in joined
    assert "前十大持仓合计" in joined
    assert "基于当前可验证的参考持仓数据" in joined


def test_etf_top_holdings_uses_partial_label_for_small_etf():
    """XLK is a real demo-portfolio holding with only 5 reference
    constituents -- must never claim "前十大持仓合计" (Top 10)."""
    app = _app()
    app.session_state["treemap_selected"] = "XLK"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "主要持仓（Top Holdings）" in joined
    assert "已显示持仓合计" in joined
    assert "前十大持仓合计" not in joined


def test_etf_top_holdings_omitted_for_etf_without_reference_data():
    """SGOV is a real demo-portfolio holding with no entry in
    core.reference.ETF_HOLDINGS -- the section must fail closed and be
    omitted entirely, never a fake/empty Top Holdings block."""
    app = _app()
    app.session_state["treemap_selected"] = "SGOV"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "证券简介（Security Profile）" in joined  # profile itself still renders
    assert "主要持仓" not in joined


def test_etf_top_holdings_never_shown_for_individual_stocks():
    app = _app()
    for ticker in ("NVDA", "AAPL"):
        app.session_state["treemap_selected"] = ticker
        app.run(timeout=20)
        assert not app.exception
        joined = "\n".join(m.value for m in app.markdown)
        assert "主要持仓" not in joined


def test_etf_top_holdings_placement_between_weight_and_key_dates():
    """Step 2A.11 hierarchy: Security Profile -> portfolio weight -> Top
    Holdings -> Key Dates -> Asset Allocation -> Portfolio Insights."""
    app = _app()
    app.session_state["treemap_selected"] = "VOO"
    app.run(timeout=20)
    joined = "\n".join(m.value for m in app.markdown)
    weight_idx = joined.index("组合占比")
    holdings_idx = joined.index("主要持仓（Top Holdings）")
    key_dates_idx = joined.index("关键关注日期")
    allocation_idx = joined.index("资产配置（Asset Allocation）")
    assert weight_idx < holdings_idx < key_dates_idx < allocation_idx


def test_etf_top_holdings_makes_no_network_call(monkeypatch):
    """Selecting an ETF must never trigger a new yfinance fetch solely for
    Top Holdings -- core.etf_holdings is a pure local dict lookup."""
    import yfinance as yf

    def _forbidden_ticker(*args, **kwargs):
        raise AssertionError("must not construct yf.Ticker for Top Holdings")

    monkeypatch.setattr(yf, "Ticker", _forbidden_ticker)
    app = _app()
    app.session_state["treemap_selected"] = "VOO"
    app.run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "主要持仓（Top Holdings）" in joined


def test_etf_top_holdings_selection_requires_no_gemini_or_anthropic_call(monkeypatch):
    import core.ai

    def _forbidden(*args, **kwargs):
        raise AssertionError("ETF Top Holdings must never call run_ai_analysis")

    monkeypatch.setattr(core.ai, "run_ai_analysis", _forbidden)
    app = _app()
    app.session_state["treemap_selected"] = "VOO"
    app.run(timeout=20)
    assert not app.exception


def test_risk_level_value_is_chinese_only_no_english_suffix():
    app = _app()
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "风险等级（Risk Level）" in joined
    for english in ("Measured", "Elevated", "High)", "(High"):
        assert english not in joined


def test_asset_allocation_uses_bonds_and_cash_like_label():
    """UI label change only -- demo portfolio holds SGOV (Fixed Income),
    so the donut legend must show the renamed bilingual label."""
    import json

    app = _app()
    assert not app.exception
    donut = next(c for c in app.get("plotly_chart") if c.key == "chart_asset_allocation")
    labels = json.loads(donut.proto.spec)["data"][0]["labels"]
    assert any("债券与现金类" in label and "Bonds & Cash-like" in label for label in labels)
    assert not any("固定收益" in label for label in labels)


def test_page3_free_plan_exists_with_zero_price_and_core_features():
    app = _app()
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "普通版 Free" in joined
    assert '<div class="pricing-price">$0</div>' in joined
    assert "永久免费" in joined
    for feature in ("投资组合概览", "Portfolio Score", "Treemap 持仓分析", "资产配置分析", "持仓集中度分析", "ETF 穿透与真实风险"):
        assert feature in joined
    assert any(b.label == "当前方案" and b.proto.disabled for b in app.button)


def test_page3_pro_plan_exists_with_displayed_price_and_features():
    app = _app()
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "Pro 高级会员" in joined
    assert "US$19.90" in joined and "/ 月" in joined
    assert "推荐" in joined and "Recommended" in joined
    for feature in ("包含普通版全部功能", "七人 AI 投资委员会", "主席综合决议", "个性化行动方案", "阶段性投资计划", "深度 AI 组合解读"):
        assert feature in joined
    assert any(b.label == "升级 Pro" for b in app.button)


def test_page3_upgrade_cta_does_not_invoke_payment():
    """The CTA must only ever show an informational message -- no Stripe,
    no checkout, no fake payment success, and no state change implying a
    purchase happened."""
    app = _app()
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "升级 Pro")
    btn.click().run(timeout=20)
    assert not app.exception
    assert any("Pro 高级会员即将开放。正式版本将支持在线订阅。Beta 测试用户可使用邀请码免费体验 Pro 功能。" in i.value for i in app.info)
    assert app.session_state["ai_unlocked"] is False
    joined = "\n".join(m.value for m in app.markdown) + "\n".join(i.value for i in app.info)
    for forbidden in ("Stripe", "支付成功", "订阅成功", "payment successful", "checkout"):
        assert forbidden not in joined


def test_page3_beta_code_stays_secondary_and_collapsed():
    app = _app()
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    assert not app.exception
    expander = next(e for e in app.expander if "邀请码" in e.label)
    assert expander.proto.expanded is False
    joined = "\n".join(m.value for m in app.markdown)
    # The pricing cards, not the beta path, must be the first thing on the page.
    assert joined.index("普通版 Free") < joined.index("邀请码")


def test_disclaimer_appears_identically_on_all_three_pages():
    disclaimer_zh = "免责声明：本系统内容由 AI 与数据模型生成，仅供信息与分析参考，不构成任何投资建议。"
    disclaimer_en = "Disclaimer: AI- and data-generated content is for informational and analytical purposes only"
    for page_label in ("1 · 投资组合概览", "2 · 真实风险与 ETF 穿透", "3 · AI 投资委员会"):
        app = _app()
        nav = app.segmented_control(key="primary_nav")
        nav.set_value(page_label).run(timeout=20)
        assert not app.exception
        joined = "\n".join(m.value for m in app.markdown)
        assert joined.count(disclaimer_zh) == 1, f"disclaimer missing/duplicated on {page_label}"
        assert disclaimer_en in joined


def test_page3_locked_by_default():
    app = _app()
    assert not app.exception
    assert app.session_state["ai_unlocked"] is False


def test_page3_pricing_copy_renders_free_vs_pro_not_single_premium():
    app = _app()
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    assert not app.exception
    joined = "\n".join(m.value for m in app.markdown)
    assert "AI 投资分析与行动方案" in joined
    assert "选择适合您的方案" in joined
    # The old single-tier "Premium" framing must be gone.
    assert "高级会员专享" not in joined
    assert any("邀请码" in e.label for e in app.expander)
    # Explicit copy-reduction from earlier rounds must still hold.
    assert "看懂投资组合只是第一步" not in joined
    assert "七人 AI 投资委员会基于您的投资组合" not in joined
    assert "AI 分析基于系统已计算的投资组合数据" not in joined
    assert "免责声明：本系统内容由 AI 与数据模型生成" in joined


def test_page3_beta_code_fails_closed_when_unconfigured(monkeypatch):
    monkeypatch.delenv("CANAM_BETA_CODES", raising=False)
    app = _app()
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    assert not app.exception
    btn = next(b for b in app.button if b.label == "验证并解锁")
    btn.click().run(timeout=20)
    assert not app.exception
    assert any("尚未开放 Pro 体验" in i.value for i in app.info)
    assert app.session_state["ai_unlocked"] is False


def test_ai_provider_failure_shows_only_generic_message_never_raw_exception(monkeypatch):
    """Diagnostic patch guard: whatever a Gemini/Anthropic failure actually
    is server-side (model, quota, network, SDK...), the public UI must keep
    showing only the generic 'AI 分析当前不可用。' string -- never the raw
    provider exception text or type name."""
    import core.ai

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated: 404 model not found: models/gemini-3.5-flash")

    monkeypatch.setattr(core.ai, "run_ai_analysis", _raise)
    app = _app()  # first run: default demo-state init happens, incl. ai_unlocked=False
    app.session_state["ai_unlocked"] = True  # then override, so the init block doesn't reset it
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception
    joined = "\n".join(i.value for i in app.info)
    assert "AI 分析当前不可用" in joined
    assert "model not found" not in joined
    assert "gemini-3.5-flash" not in joined
    assert "RuntimeError" not in joined
    assert app.session_state["ai_error"] == "RuntimeError"


def test_final_503_shows_busy_message_not_generic_unavailable_message(monkeypatch):
    """Gemini reliability patch, TESTS REQUIRED #9, #11: when the final
    propagated provider failure (after all primary retries and the bounded
    fallback retry are exhausted -- see core.ai.GeminiProvider) carries a
    confirmed HTTP 503, the UI must show the specific busy-retry message
    instead of the generic unavailable one, and must still never leak the
    raw exception text/type."""
    import core.ai
    from google.genai import errors as genai_errors

    def _raise(*args, **kwargs):
        raise genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})

    monkeypatch.setattr(core.ai, "run_ai_analysis", _raise)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception
    joined = "\n".join(i.value for i in app.info)
    assert "AI 服务当前繁忙，请稍后重试。" in joined
    assert "AI 分析当前不可用" not in joined
    assert "high demand" not in joined
    assert "UNAVAILABLE" not in joined
    assert "ServerError" not in joined
    assert app.session_state["ai_error_status"] == 503


def test_final_non_503_error_keeps_generic_unavailable_message(monkeypatch):
    """TESTS REQUIRED #10: a final failure that is NOT a confirmed 503 (here,
    a permanent 400 ClientError -- e.g. an invalid/misconfigured model) must
    keep showing the existing generic message, never the 503-specific busy
    message."""
    import core.ai
    from google.genai import errors as genai_errors

    def _raise(*args, **kwargs):
        raise genai_errors.ClientError(400, {"error": {"code": 400, "message": "invalid argument", "status": "INVALID_ARGUMENT"}})

    monkeypatch.setattr(core.ai, "run_ai_analysis", _raise)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception
    joined = "\n".join(i.value for i in app.info)
    assert "AI 分析当前不可用" in joined
    assert "AI 服务当前繁忙" not in joined
    assert "invalid argument" not in joined
    assert "INVALID_ARGUMENT" not in joined
    assert "ClientError" not in joined
    assert app.session_state["ai_error_status"] == 400


def test_action_plan_renders_decision_first_sections_without_exception(monkeypatch):
    """Step 2A.10 human-acceptance smoke test for the decision-first Page 3
    hierarchy: a valid mocked committee result renders Chairman Decision,
    Current Action Summary (do now/do not now only), Key Security Actions,
    and Key Events & Reassessment Triggers -- with no exception, using only
    tickers that are real holdings in the demo portfolio the app starts
    with. Strategy Now, Top Actions, Action Timeline, and the Execution
    Checklist are deliberately no longer rendered (see the Step 2A.10
    repetition rule below)."""
    import core.ai
    from tests.test_ai import VALID

    parsed = core.ai.CommitteeResult.model_validate(VALID)

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    joined = "\n".join(m.value for m in app.markdown)
    assert "主席决策" in joined
    assert "当前行动结论" in joined
    assert "现在做" in joined and "现在不做" in joined
    assert "重点持仓行动" in joined
    assert "NVDA" in joined
    assert "关键事件与重新评估条件" in joined
    # Deliberately no longer rendered in the primary UI (see the repetition
    # rule test below) -- the schema still generates these fields.
    assert "当前总策略" not in joined
    assert "本阶段最重要的几件事" not in joined
    assert "行动时间线" not in joined
    assert "执行清单" not in joined
    # The old standalone modules must be gone from the rendered page.
    assert "组合目标迁移" not in joined
    assert "维持不动的仓位" not in joined
    assert "情景应对" not in joined
    assert "关键决策日历" not in joined


def test_action_plan_with_empty_security_actions_renders_without_exception_and_hides_header(monkeypatch):
    """Step 2A.3 P1-1 follow-up (D, E), updated for Step 2A.10's decision-
    first hierarchy: a genuinely no-action portfolio may still return
    security_actions=[] (see core.ai.ActionPlan). The page must not crash,
    and must not show the "重点持仓行动（Key Security Actions）" section
    label/heading with nothing under it -- every other decision-first
    section still renders normally."""
    import json

    import core.ai
    from tests.test_ai import VALID

    action_plan = json.loads(json.dumps(VALID["action_plan"]))
    action_plan["security_actions"] = []
    parsed = core.ai.CommitteeResult.model_validate({**VALID, "action_plan": action_plan})

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    joined = "\n".join(m.value for m in app.markdown)
    assert "重点持仓行动" not in joined
    # Every other decision-first section still renders.
    assert "主席决策" in joined
    assert "当前行动结论" in joined
    assert "关键事件与重新评估条件" in joined


def test_chairman_decision_renders_before_action_plan_and_committee_details(monkeypatch):
    """Step 2A.10: Chairman Decision must be the first major result on Page
    3, followed by the Action Plan and Key Security Actions, with the
    collapsed committee-details expander last -- the user should not have
    to scroll through six specialist cards before reaching the decision."""
    import core.ai
    from tests.test_ai import VALID

    parsed = core.ai.CommitteeResult.model_validate(VALID)

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    # The expander's own label lives on the widget, not in app.markdown --
    # use a specialist role label (rendered only inside the expander body)
    # as the proxy for "committee details content" in the ordering check.
    expander = next(e for e in app.expander if "投资委员会详细意见" in e.label)
    assert expander.proto.expanded is False
    joined = "\n".join(m.value for m in app.markdown)
    chairman_idx = joined.index("主席决策")
    action_plan_idx = joined.index("条件式行动方案")
    do_now_idx = joined.index("现在做（Do Now）")
    key_security_idx = joined.index("重点持仓行动")
    key_events_idx = joined.index("关键事件与重新评估条件")
    committee_content_idx = joined.index("宏观与市场")
    assert chairman_idx < action_plan_idx < do_now_idx < key_security_idx < key_events_idx < committee_content_idx


def test_committee_details_collapsed_by_default_and_contains_specialist_content(monkeypatch):
    """Step 2A.10: the six specialist cards, Majority View, and Main Concern
    move into a collapsed-by-default expander below the action content --
    supporting reasoning, not the first thing the user reads. The seven-
    member architecture (six specialists + Chairman) and every specialist
    role are preserved, only their position on the page changes."""
    import core.ai
    from tests.test_ai import VALID

    parsed = core.ai.CommitteeResult.model_validate(VALID)

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    expander = next(e for e in app.expander if "投资委员会详细意见" in e.label)
    assert expander.proto.expanded is False
    inside = "\n".join(m.value for m in expander.markdown)
    assert "七人投资委员会" in inside
    for role_label in ["宏观与市场", "组合结构", "风险（Risk）", "估值与数据", "税务与账户", "行动与再平衡"]:
        assert role_label in inside
    assert "多数意见" in inside and "维持核心配置并优化结构" in inside
    assert "主要关切" in inside and "集中度需要持续观察" in inside
    # Chairman Decision itself is NOT inside this expander -- it already
    # rendered above, before the expander.
    assert "主席决策" not in inside


def test_ai_suggested_target_range_is_explicitly_labeled(monkeypatch):
    """Step 2A.10 semantic hardening: any AI-generated target must be
    explicitly labeled as an AI suggested range, never presented as a
    deterministic optimizer output -- there is no target-allocation
    optimizer, correlation model, or destination-recommendation engine in
    this project."""
    import json

    import core.ai
    from tests.test_ai import VALID

    action_plan = json.loads(json.dumps(VALID["action_plan"]))
    action_plan["security_actions"][0]["target"] = "8%-10%"
    parsed = core.ai.CommitteeResult.model_validate({**VALID, "action_plan": action_plan})

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    joined = "\n".join(m.value for m in app.markdown)
    assert "AI 建议目标区间" in joined
    assert "AI Suggested Range" in joined
    assert "8%-10%" in joined
    # Never implied as a deterministic optimizer/allocation target -- and
    # the old bare "目标" label (with no AI-suggested qualifier) is gone.
    assert "<b>目标</b>" not in joined
    for forbidden in ("最优配置", "确定性目标", "优化器推荐"):
        assert forbidden not in joined


def test_trade_impact_preview_renders_for_reduce_action_with_reduction_pct(monkeypatch):
    """Step 2A human-acceptance smoke test: a REDUCE security_action with
    position_reduction_pct set renders the compact Trade Impact Preview
    under the demo portfolio's real NVDA holding (55 shares @ $112 avg cost,
    per core/demo.py) -- no exception, no tax-payable wording, no invented
    destination security, proceeds explicitly held as cash."""
    import json

    import core.ai
    from tests.test_ai import VALID

    action_plan = json.loads(json.dumps(VALID["action_plan"]))
    action_plan["security_actions"][0] = {
        "ticker": "NVDA", "action": "REDUCE", "priority": "高",
        "target": "建议目标：减少约10%当前仓位", "trigger": "若NVDA重新回到近期相对强势区间",
        "reason": "直接权重偏高，具体交易影响见下方本地确定性预览。",
        "position_reduction_pct": 10.0,
    }
    parsed = core.ai.CommitteeResult.model_validate({**VALID, "action_plan": action_plan})

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    joined = "\n".join(m.value for m in app.markdown)
    assert "交易影响预览" in joined
    assert "可执行约 5 股" in joined  # floor(55 * 10%) = 5
    assert "预计已实现盈亏" in joined
    assert "直接权重" in joined and "已识别穿透暴露" in joined and "Top-5 集中度" in joined
    assert "现金" in joined  # cash-proceeds assumption stated
    # Must never appear anywhere on the page: tax-payable claims, TLH, or an
    # invented reinvestment destination security.
    for forbidden in ("应缴税", "税款", "税负", "CRA", "损失收割", "买入 VOO", "买入VOO", "转为 VOO", "转为VOO"):
        assert forbidden not in joined


def test_trade_impact_preview_is_marked_conditional_for_reduce_on_rebound(monkeypatch):
    """Step 2A.2 human-acceptance smoke test (I/J): a REDUCE_ON_REBOUND
    security_action with position_reduction_pct set must still render the
    same deterministic before/after numbers, but as an explicitly
    CONDITIONAL preview -- never with immediate-execution wording."""
    import json

    import core.ai
    from tests.test_ai import VALID

    action_plan = json.loads(json.dumps(VALID["action_plan"]))
    action_plan["security_actions"][0] = {
        "ticker": "NVDA", "action": "REDUCE_ON_REBOUND", "priority": "高",
        "target": "AI建议减持直接持仓约15%-20%，使穿透暴露控制在合理区间",
        "trigger": "若NVDA重新回到近期相对强势区间",
        "reason": "直接权重偏高，具体交易影响见下方本地确定性预览。",
        "position_reduction_pct": 15.0,
    }
    parsed = core.ai.CommitteeResult.model_validate({**VALID, "action_plan": action_plan})

    def _fake_run_ai_analysis(result, **kwargs):
        return parsed, {"provider": "gemini", "model": "gemini-3.6-flash", "cache_hit": False}

    monkeypatch.setattr(core.ai, "run_ai_analysis", _fake_run_ai_analysis)
    app = _app()
    app.session_state["ai_unlocked"] = True
    nav = app.segmented_control(key="primary_nav")
    nav.set_value("3 · AI 投资委员会").run(timeout=20)
    btn = next(b for b in app.button if b.label == "启动 AI 投资委员会")
    btn.click().run(timeout=20)
    assert not app.exception

    joined = "\n".join(m.value for m in app.markdown)
    assert "条件触发后的交易影响预览" in joined  # I: explicitly marked conditional
    assert "Conditional Trade Impact Preview" in joined
    assert "仅表示触发条件满足后的情景预览，不代表当前立即执行" in joined
    assert "可执行约 8 股" in joined  # floor(55 * 15%) = 8, same deterministic math as REDUCE
    # J: must never phrase this as an instruction to execute right now (the
    # disclaimer itself legitimately contains "...不代表当前立即执行" as a
    # negation, so check for the affirmative phrasing specifically).
    for forbidden in ("现在立刻减持", "请立即执行", "立刻卖出", "请求：当前仓位减少约"):
        assert forbidden not in joined


def test_runtime_version_logs_short_source_version(monkeypatch, caplog):
    """Diagnostic patch: with SOURCE_VERSION set (as Streamlit Cloud sets it
    to the deployed commit hash), the startup log must report only a short
    (12-char) prefix -- enough to confirm the deployed commit without
    dumping the full env value."""
    monkeypatch.setenv("SOURCE_VERSION", "59167ac46d4528ca50dfeccf550d33e3a9645a28")
    with caplog.at_level("INFO", logger="canam.app"):
        _app()
    messages = [r.message for r in caplog.records if r.name == "canam.app"]
    assert any(m == "CanAm runtime version: 59167ac46d45" for m in messages)


def test_runtime_version_logs_unknown_when_source_version_absent(monkeypatch, caplog):
    monkeypatch.delenv("SOURCE_VERSION", raising=False)
    with caplog.at_level("INFO", logger="canam.app"):
        _app()
    messages = [r.message for r in caplog.records if r.name == "canam.app"]
    assert any(m == "CanAm runtime version: unknown" for m in messages)


def test_runtime_version_log_never_contains_secrets(monkeypatch, caplog):
    monkeypatch.setenv("SOURCE_VERSION", "59167ac46d4528ca50dfeccf550d33e3a9645a28")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySECRETVALUESHOULDNEVERAPPEAR")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRETVALUESHOULDNEVERAPPEAR")
    with caplog.at_level("INFO", logger="canam.app"):
        _app()
    messages = [r.message for r in caplog.records if r.name == "canam.app"]
    joined = "\n".join(messages)
    assert "SECRETVALUESHOULDNEVERAPPEAR" not in joined


def test_apply_holdings_commits_exactly_the_frame_it_is_given():
    """Guards 'editing imported quantity -> analyze uses edited quantity':
    the single Analyze button always calls _apply_holdings(edited), and
    _apply_holdings must be a pure passthrough of whatever frame it receives
    -- never a second, independently-tracked data source."""
    import sys
    sys.path.insert(0, str(APP_PATH.parent))
    from app import _holdings_from_frame

    edited_frame = pd.DataFrame([{"Ticker": "NVDA", "Quantity": 99, "Average Cost": None}])
    holdings = _holdings_from_frame(edited_frame)
    assert holdings[0].ticker == "NVDA"
    assert holdings[0].quantity == 99
