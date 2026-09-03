from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.ai import ProviderUnavailable, run_ai_analysis
from core.analytics import analyze, build_snapshot
from core.auth import configured_beta_codes, validate_beta_code
from core.demo import DEMO_HOLDINGS, DEMO_QUOTES
from core.importers import auto_map_columns, build_preview, missing_required_fields, read_portfolio_file, template_csv_bytes
from core.market import fetch_quotes, fetch_usd_cad_rate
from core.models import ACCOUNT_TYPES, HoldingInput
from ui import brand_header, committee_view, inject_theme, overview, page_disclaimer, risk_page

load_dotenv()
st.set_page_config(page_title="加美通 AI 投资分析", page_icon="◈", layout="wide", initial_sidebar_state="auto")
inject_theme()

HOLDINGS_COLUMNS = ["Ticker", "Quantity", "Average Cost"]


def _demo_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"Ticker": h.ticker, "Quantity": h.quantity, "Average Cost": h.average_cost}
        for h in DEMO_HOLDINGS
    ], columns=HOLDINGS_COLUMNS)


def _holdings_from_frame(frame: pd.DataFrame) -> list[HoldingInput]:
    return [
        HoldingInput(str(r["Ticker"]), float(r["Quantity"]), float(r["Average Cost"]) if pd.notna(r.get("Average Cost")) else None)
        for _, r in frame.iterrows()
    ]


def _ticker_set(frame: pd.DataFrame) -> set[str]:
    return {str(t).strip().upper().replace(".", "-") for t in frame["Ticker"] if str(t).strip()}


def _apply_holdings(frame: pd.DataFrame) -> bool:
    """The single commit path: manual edits and file imports both end here.
    This is the only place that writes st.session_state.holdings, and it
    always fetches quotes for exactly those holdings in the same step, so
    holdings and quotes can never drift apart (the P0 data-mismatch class)."""
    try:
        holdings = _holdings_from_frame(frame)
        with st.spinner("正在获取已验证市场数据…（Fetching verified market data）"):
            st.session_state.quotes = fetch_quotes([h.ticker for h in holdings])
            st.session_state.usd_cad = fetch_usd_cad_rate()
        st.session_state.holdings = frame
        st.session_state.analyzed_tickers = _ticker_set(frame)
        st.session_state.is_demo = False
        st.session_state.ai_result = None
        st.session_state.treemap_selected = None
        return True
    except Exception as exc:
        st.error(f"无法完成分析（Portfolio could not be analyzed）：{exc}")
        return False


if "holdings" not in st.session_state:
    st.session_state.holdings = _demo_frame()
    st.session_state.quotes = DEMO_QUOTES
    st.session_state.analyzed_tickers = _ticker_set(st.session_state.holdings)
    st.session_state.is_demo = True
    st.session_state.cash = 2500.0
    st.session_state.account_currency = "USD"
    st.session_state.account_type = "Taxable"
    st.session_state.usd_cad = 1.36
    st.session_state.ai_unlocked = False
    st.session_state.ai_result = None

with st.sidebar:
    st.markdown("### 投资组合（Portfolio）")
    with st.expander("导入持仓（CSV/XLSX）", expanded=False):
        uploaded = st.file_uploader("上传 CSV 或 XLSX 文件", type=["csv", "xlsx", "xls"], key="portfolio_upload")
        st.download_button(
            "下载模板", data=template_csv_bytes(), file_name="portfolio_template.csv",
            mime="text/csv", key="download_template",
        )
        if uploaded is not None:
            # Parse once per uploaded file (Streamlit's own file_id), not on every
            # rerun, and populate the canonical table at most once per file so a
            # later rerun (e.g. a page switch) never re-clobbers the user's edits.
            raw_key, applied_key = f"_import_raw_{uploaded.file_id}", f"_import_applied_{uploaded.file_id}"
            if raw_key not in st.session_state:
                try:
                    st.session_state[raw_key] = read_portfolio_file(uploaded.getvalue(), uploaded.name)
                except Exception as exc:
                    st.session_state[raw_key] = exc
            raw = st.session_state[raw_key]
            if isinstance(raw, Exception):
                st.error(f"无法读取文件（Could not read file）：{raw}")
            else:
                mapping = auto_map_columns(raw)
                missing = missing_required_fields(mapping)
                if missing:
                    st.caption("部分必需字段无法自动识别，请手动选择对应列：")
                    field_labels = {"ticker": "代码列（Ticker）", "quantity": "数量列（Quantity）"}
                    columns = list(raw.columns)
                    for field in missing:
                        choice = st.selectbox(field_labels[field], ["请选择…"] + columns, key=f"import_map_{field}")
                        if choice != "请选择…":
                            mapping[field] = choice
                if not missing_required_fields(mapping):
                    if applied_key not in st.session_state:
                        st.session_state.holdings = build_preview(raw, mapping)
                        st.session_state.is_demo = False
                        st.session_state.ai_result = None
                        st.session_state.treemap_selected = None
                        st.session_state[applied_key] = True
                        st.rerun()
                    st.caption(f"已导入 {len(st.session_state.holdings)} 项持仓")
    edited = st.data_editor(
        st.session_state.holdings, num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Ticker": st.column_config.TextColumn("代码（Ticker）", required=True),
            "Quantity": st.column_config.NumberColumn("数量（Quantity）", min_value=0.0001, format="%.4f", required=True),
            "Average Cost": st.column_config.NumberColumn("平均成本（Average Cost）", min_value=0.0, format="%.2f"),
        }, key="portfolio_editor",
    )
    st.session_state.account_type = st.selectbox(
        "账户类型（Account Type）", list(ACCOUNT_TYPES),
        format_func=lambda v: "TFSA（免税账户）" if v == "TFSA" else "Taxable（应税账户）",
        index=0 if st.session_state.account_type == "TFSA" else 1, key="account_type_select",
    )
    st.session_state.cash = st.number_input("现金（Cash）", min_value=0.0, value=float(st.session_state.cash), step=500.0, key="cash_input")
    st.session_state.account_currency = st.selectbox("账户货币（Account Currency）", ["USD", "CAD"], index=0 if st.session_state.account_currency == "USD" else 1, key="account_currency_select")
    if st.button("开始分析", type="primary", width="stretch", key="analyze"):
        _apply_holdings(edited)
    if st.button("恢复示例组合", width="stretch", key="restore_demo"):
        st.session_state.holdings = _demo_frame()
        st.session_state.quotes = DEMO_QUOTES
        st.session_state.analyzed_tickers = _ticker_set(st.session_state.holdings)
        st.session_state.is_demo = True
        st.session_state.cash = 2500.0
        st.session_state.account_currency = "USD"
        st.session_state.account_type = "Taxable"
        st.session_state.usd_cad = 1.36
        st.session_state.ai_result = None
        st.session_state.treemap_selected = None
        st.rerun()
    st.caption("持仓只在当前本机会话中处理。只有解锁并明确启动 AI 后，最小化事实包才会发送给所选 AI 提供商。")

# Invariant: st.session_state.holdings is only ever written together with a
# matching quote fetch, and analyzed_tickers records exactly which ticker
# set that fetch covered (inside _apply_holdings / demo restore). If an
# import just populated the table but the user has not clicked "开始分析"
# yet, the displayed tickers no longer match analyzed_tickers — show that
# plainly instead of computing a result from stale/mismatched quotes. A
# plain "is this ticker present in quotes" check would miss the case where
# the new tickers happen to overlap with the previous portfolio's quotes.
if _ticker_set(st.session_state.holdings) != st.session_state.analyzed_tickers:
    brand_header(st.session_state.is_demo)
    st.info("持仓已更新，请点击左侧「开始分析」以获取最新市场数据。")
    st.stop()

holdings = _holdings_from_frame(st.session_state.holdings)

# Fail-closed ticker identity gate: a ticker that resolved to a quote for
# an unverified or mismatched security (e.g. bare "FINN" colliding with
# an unrelated U.S. OTC stock) must never silently enter valuation as a
# zero/dropped position. Block the deterministic analytics pages entirely
# until the user fixes the ticker, per the compact message from
# core.ticker_resolution / core.market.fetch_quotes.
identity_errors = sorted({
    quote.status.split("identity:", 1)[1]
    for ticker in {h.ticker.strip().upper() for h in holdings}
    if (quote := st.session_state.quotes.get(ticker)) is not None and quote.status.startswith("identity:")
})
if identity_errors:
    brand_header(st.session_state.is_demo)
    for message in identity_errors:
        st.error(message)
    st.stop()

try:
    snapshot = build_snapshot(
        holdings, st.session_state.quotes, cash=st.session_state.cash,
        account_currency=st.session_state.account_currency, usd_cad=st.session_state.usd_cad,
        account_type=st.session_state.account_type,
    )
    result = analyze(snapshot)
except Exception as exc:
    brand_header(st.session_state.is_demo)
    st.error(f"无法形成可靠估值：{exc}")
    st.stop()

brand_header(st.session_state.is_demo)
page = st.segmented_control("Primary navigation", ["1 · 投资组合概览", "2 · 真实风险与 ETF 穿透", "3 · AI 投资委员会"], default="1 · 投资组合概览", label_visibility="collapsed", key="primary_nav")

if page == "1 · 投资组合概览":
    overview(result)
elif page == "2 · 真实风险与 ETF 穿透":
    risk_page(result)
else:
    st.markdown("## AI 投资分析与行动方案")
    if not st.session_state.ai_unlocked:
        st.markdown('<div class="page-lead">选择适合您的方案</div>', unsafe_allow_html=True)
        free_features = ["投资组合概览", "Portfolio Score", "Treemap 持仓分析", "资产配置分析", "持仓集中度分析", "ETF 穿透与真实风险"]
        pro_features = ["包含普通版全部功能", "七人 AI 投资委员会", "主席综合决议", "个性化行动方案", "阶段性投资计划", "深度 AI 组合解读"]
        free_items = "".join(f'<div class="pricing-feature"><span class="pricing-check">✓</span>{f}</div>' for f in free_features)
        pro_items = "".join(f'<div class="pricing-feature"><span class="pricing-check">✓</span>{f}</div>' for f in pro_features)

        col_free, col_pro = st.columns(2, gap="large")
        with col_free:
            st.markdown(
                '<div class="pricing-card">'
                '<div class="pricing-name">普通版 Free</div>'
                '<div class="pricing-price">$0</div>'
                '<div class="pricing-period">永久免费</div>'
                f'<div class="pricing-features">{free_items}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.button("当前方案", disabled=True, width="stretch", key="current_plan_free")
        with col_pro:
            st.markdown(
                '<div class="pricing-card pricing-card-pro">'
                '<span class="pricing-badge">推荐 · Recommended</span>'
                '<div class="pricing-name">Pro 高级会员</div>'
                '<div class="pricing-price">US$19.90<span class="pricing-price-unit"> / 月</span></div>'
                f'<div class="pricing-features">{pro_items}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("升级 Pro", type="primary", width="stretch", key="upgrade_premium"):
                st.info("Pro 高级会员即将开放。正式版本将支持在线订阅。Beta 测试用户可使用邀请码免费体验 Pro 功能。")
        st.caption("无需长期合约，可随时取消。No long-term commitment. Cancel anytime.")
        with st.expander("已有邀请码？输入邀请码 / Beta Code", expanded=False):
            st.markdown('<b>Pro 高级会员</b><br><small style="color:#58708D">七人 AI 投资委员会属于 Pro 功能。正式版本上线后可通过订阅解锁。当前测试用户可输入邀请码提前体验。</small>', unsafe_allow_html=True)
            code = st.text_input("邀请码 / Beta Code", type="password", key="beta_code_input")
            if st.button("验证并解锁", key="unlock_ai"):
                if not configured_beta_codes():
                    st.info("当前测试环境尚未开放 Pro 体验。")
                elif validate_beta_code(code):
                    st.session_state.ai_unlocked = True
                    st.rerun()
                else:
                    st.error("访问码无效。")
    else:
        st.markdown('<div class="quiet-card"><b>Pro 已解锁</b><br><span style="color:#58708D">不会自动调用 API。点击下方按钮后才会发送最小化事实包。</span></div>', unsafe_allow_html=True)
        if st.button("启动 AI 投资委员会", type="primary", key="run_committee"):
            try:
                with st.spinner("投资委员会正在审阅确定性事实…（Reviewing canonical facts）"):
                    st.session_state.ai_result, st.session_state.ai_meta = run_ai_analysis(result)
            except Exception as exc:
                st.session_state.ai_result = None
                st.session_state.ai_error = type(exc).__name__
        if st.session_state.ai_result:
            committee_view(st.session_state.ai_result)
        elif st.session_state.get("ai_error") or not (os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            st.info("AI 分析当前不可用。")
    page_disclaimer()
