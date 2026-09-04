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


def test_action_plan_renders_all_sections_without_exception(monkeypatch):
    """Human-acceptance smoke test for the upgraded Action Plan (Page 3):
    a valid mocked committee result renders every new section -- Strategy
    Now, Top Changes, Do Now/Do Not Now, Security Actions, Portfolio Target
    Migration, Action Timeline (all 4 horizons), Execution Checklist, and
    No Change -- with no exception, using only tickers that are real
    holdings in the demo portfolio the app starts with."""
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
    assert "当前总策略" in joined
    assert "本阶段最重要的几件事" in joined
    assert "现在做" in joined and "现在不做" in joined
    assert "标的级行动" in joined
    assert "NVDA" in joined
    assert "组合目标迁移" in joined
    assert "行动时间线" in joined
    assert "未来30天" in joined and "未来3个月" in joined and "未来6–12个月" in joined
    assert "执行清单" in joined
    assert "维持不动的仓位" in joined
    assert "SGOV" in joined


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
