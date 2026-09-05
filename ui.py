from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.analytics import treemap_rows
from core.models import AnalyticsResult
from core.reference import ASSET_CLASS_LABELS_EN, ASSET_CLASS_LABELS_ZH
from core.security_profile import resolve_security_profile
from core.trade_preview import build_trade_impact_preview

# Low-saturation, blue-first palette: dark muted navy through soft blue-gray.
# No electric/royal/cobalt blue — hierarchy comes from value/lightness, not saturation.
BLUE = ["#16283B", "#24405C", "#3B6690", "#6A8CA3", "#9FB8C9", "#D6E2EA"]
RISK = "#C92A2A"
RISK_LEVEL_LABELS = {"High": "高", "Elevated": "偏高", "Measured": "适中"}
DONUT_COLORS = {"Equity": "#3B6690", "Fixed Income": "#24405C", "Cash": "#16283B", "Other": "#6A8CA3"}


def inject_theme() -> None:
    st.markdown("""
    <style>
    .stApp { background: #F7F9FC; }
    .block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 4rem; }
    h1,h2,h3 { color:#102A43; letter-spacing:-.025em; }
    [data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E4EBF5; }
    [data-testid="stMetric"] { background:transparent; border-left:1px solid #D9E4F2; padding-left:1rem; }
    .brand-kicker { color:#3B6690; font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .brand-title { color:#102A43; font-size:2.2rem; font-weight:760; letter-spacing:-.045em; margin:.15rem 0 0; }
    .brand-en { color:#58708D; font-size:1rem; margin:0 0 1.5rem; }
    .section-label { color:#3B6690; font-size:.74rem; font-weight:720; letter-spacing:.12em; text-transform:uppercase; margin-top:1rem; }
    .quiet-card { background:#FFF; border:1px solid #E2EAF4; border-radius:16px; padding:1.1rem 1.2rem; }
    .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0; margin:.25rem 0 1rem; }
    .metric-item { border-left:1px solid #D9E4F2; padding:.2rem 1rem .5rem; min-width:0; }
    .metric-label { color:#40566F; font-size:.82rem; }
    .metric-value { color:#102A43; font-size:2rem; line-height:1.2; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:.3rem; }
    .risk-number,.risk-badge { color:#C92A2A; font-weight:750; }
    .member { min-height:126px; }
    .member-role { color:#3B6690; font-size:.76rem; font-weight:700; }
    .member-stance { color:#102A43; font-weight:700; margin:.3rem 0; }
    .member-copy { color:#58708D; font-size:.9rem; line-height:1.45; }
    .true-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:1rem; }
    .member-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin-bottom:1rem; }
    .plan-step { border-left:3px solid #3B6690; padding:.2rem 0 .2rem 1rem; margin:.65rem 0 1.2rem; }
    .demo-badge { display:inline-block;background:#E4EBF2;color:#24405C;border-radius:99px;padding:.25rem .65rem;font-size:.75rem;font-weight:700; }
    .page-lead { color:#102A43; font-size:1.05rem; font-weight:650; margin:.2rem 0 1rem; }
    .rank-list,.etf-list { display:flex; flex-direction:column; gap:.7rem; margin-top:.5rem; }
    .rank-row { display:grid; grid-template-columns:1.3rem 3.4rem 1fr 3.6rem; align-items:center; gap:.6rem; }
    .rank-num { color:#8AA0B8; font-size:.78rem; font-weight:700; text-align:right; }
    .rank-ticker { color:#102A43; font-weight:700; font-size:.86rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .rank-track,.etf-track { background:#EAF0F8; border-radius:99px; height:6px; overflow:hidden; display:flex; }
    .rank-fill { background:#3B6690; height:100%; border-radius:99px; }
    .rank-pct,.etf-total { color:#102A43; font-size:.86rem; text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
    .etf-row { display:grid; grid-template-columns:3.6rem 1fr 4.4rem; align-items:center; gap:.6rem; }
    .etf-ticker { color:#102A43; font-weight:700; font-size:.86rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .etf-track { height:9px; }
    .etf-direct { background:#24405C; height:100%; }
    .etf-indirect { background:#9FB8C9; height:100%; }
    .etf-legend { display:flex; gap:1.1rem; font-size:.76rem; color:#58708D; margin:.2rem 0 .7rem; }
    .etf-legend i { display:inline-block; width:.62rem; height:.62rem; border-radius:2px; margin-right:.35rem; vertical-align:middle; }
    .page-disclaimer { color:#8AA0B8; font-size:.72rem; line-height:1.5; margin-top:2rem; }
    .pricing-card { background:#FFF; border:1px solid #E2EAF4; border-radius:16px; padding:1.6rem 1.5rem; height:100%; }
    .pricing-card-pro { border:1.5px solid #3B6690; }
    .pricing-badge { display:inline-block; background:#3B6690; color:#FFF; font-size:.7rem; font-weight:700; letter-spacing:.03em; padding:.25rem .7rem; border-radius:99px; margin-bottom:.7rem; }
    .pricing-name { color:#102A43; font-weight:700; font-size:1.05rem; }
    .pricing-price { color:#102A43; font-size:1.9rem; font-weight:760; margin-top:.5rem; }
    .pricing-price-unit { color:#58708D; font-size:.92rem; font-weight:500; }
    .pricing-period { color:#58708D; font-size:.82rem; margin-top:.15rem; }
    .pricing-features { display:flex; flex-direction:column; gap:.6rem; margin:1.1rem 0 1.2rem; }
    .pricing-feature { display:flex; gap:.55rem; align-items:flex-start; color:#102A43; font-size:.86rem; line-height:1.4; }
    .pricing-check { color:#3B6690; font-weight:700; flex:0 0 auto; }
    /* --- Action Plan (Page 3) only --- */
    .ap-donow-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin:.7rem 0; }
    .ap-list { margin:.4rem 0 0; padding:0; list-style:none; display:flex; flex-direction:column; gap:.45rem; }
    .ap-list li { font-size:.88rem; line-height:1.4; color:#102A43; padding-left:1.1rem; position:relative; }
    .ap-do-now li:before { content:"✓"; color:#3B6690; position:absolute; left:0; font-weight:700; }
    .ap-do-not li:before { content:"✕"; color:#8AA0B8; position:absolute; left:0; font-weight:700; }
    .ap-priority { display:inline-block; border-radius:99px; padding:.12rem .6rem; font-size:.72rem; font-weight:700; }
    .ap-priority-high { background:#FBE3E3; color:#C92A2A; }
    .ap-priority-medhigh { background:#FCEEDC; color:#B5651D; }
    .ap-priority-med { background:#EEF4FF; color:#3B6690; }
    .ap-priority-low { background:#EAF0F8; color:#58708D; }
    .ap-security-card { background:#FFF; border:1px solid #E2EAF4; border-radius:14px; padding:.9rem 1.05rem; margin:.55rem 0; }
    .ap-security-head { display:flex; align-items:center; justify-content:space-between; gap:.6rem; }
    .ap-security-ticker { color:#102A43; font-weight:760; font-size:1rem; }
    .ap-security-action { color:#3B6690; font-weight:700; font-size:.86rem; margin-top:.1rem; }
    .ap-security-meta { color:#58708D; font-size:.82rem; line-height:1.45; margin-top:.4rem; }
    .ap-security-meta b { color:#40566F; }
    .ap-timeline-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin:.6rem 0; }
    .ap-timeline-card { background:#FFF; border:1px solid #E2EAF4; border-radius:14px; padding:.9rem 1rem; }
    .ap-timeline-title { color:#3B6690; font-size:.76rem; font-weight:720; letter-spacing:.06em; }
    .ap-timeline-objective { color:#102A43; font-weight:700; font-size:.88rem; margin:.3rem 0 .5rem; line-height:1.4; }
    .ap-timeline-card ul.ap-list li { font-size:.82rem; color:#40566F; }
    .ap-timeline-card ul.ap-list li:before { content:"–"; color:#9FB8C9; }
    .ap-timeline-trigger { color:#8AA0B8; font-size:.78rem; margin-top:.5rem; line-height:1.4; }
    .ap-calendar-row { display:grid; grid-template-columns:6.2rem 1fr 4.4rem; gap:.7rem; align-items:start; padding:.5rem 0; border-top:1px solid #EEF2F8; }
    .ap-calendar-row:first-child { border-top:none; }
    .ap-calendar-date { color:#58708D; font-size:.8rem; font-variant-numeric:tabular-nums; }
    .ap-calendar-event { color:#102A43; font-size:.86rem; font-weight:700; }
    .ap-calendar-reassess { color:#58708D; font-size:.8rem; margin-top:.15rem; }
    .ap-checklist { margin:.4rem 0 0; padding:0; list-style:none; display:flex; flex-direction:column; gap:.5rem; }
    .ap-checklist li { font-size:.86rem; color:#102A43; padding-left:1.3rem; position:relative; line-height:1.4; }
    .ap-checklist li:before { content:"☐"; color:#3B6690; position:absolute; left:0; }
    @media(max-width: 720px) {
      .block-container { padding:1rem .85rem 3rem; }
      .brand-title { font-size:1.75rem; }
      [data-testid="stMetric"] { padding-left:.65rem; }
      .metric-grid { grid-template-columns:1fr; gap:.7rem; }
      .true-grid,.member-grid { grid-template-columns:1fr; }
      .rank-row { grid-template-columns:1.1rem 2.8rem 1fr 3.2rem; }
      .etf-row { grid-template-columns:2.9rem 1fr 3.8rem; }
      .ap-donow-grid,.ap-timeline-grid { grid-template-columns:1fr; }
    }
    @media(min-width:721px) and (max-width:1000px) {
      .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); row-gap:.8rem; }
      .true-grid,.member-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .ap-timeline-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    </style>
    """, unsafe_allow_html=True)


def brand_header(is_demo: bool) -> None:
    demo = '<span class="demo-badge">示例数据（Demo Data）</span>' if is_demo else ""
    st.markdown(f'<div class="brand-kicker">投资组合智能（Portfolio Intelligence）{demo}</div><div class="brand-title">加美通 AI 投资分析</div><div class="brand-en">CanAm AI Investment Analytics</div>', unsafe_allow_html=True)


def page_disclaimer() -> None:
    """The single canonical source for the footer disclaimer shared by
    all three pages -- one call site, so the wording can never drift
    between them."""
    st.markdown(
        '<div class="page-disclaimer">免责声明：本系统内容由 AI 与数据模型生成，仅供信息与分析参考，不构成任何投资建议。'
        '<br>Disclaimer: AI- and data-generated content is for informational and analytical purposes only '
        'and does not constitute investment advice.</div>',
        unsafe_allow_html=True,
    )


def _security_profile_card(ticker: str, weight: float | None) -> None:
    st.markdown('<div class="section-label">证券简介（Security Profile）</div>', unsafe_allow_html=True)
    profile = resolve_security_profile(ticker)
    if profile is None:
        st.markdown(f'<div class="quiet-card"><b>{ticker}</b><br><small>暂无详细证券介绍</small></div>', unsafe_allow_html=True)
        return
    weight_line = f'<br><br><small>组合占比</small><br><b style="font-size:1.1rem">{weight:.1%}</b>' if weight is not None else ""
    st.markdown(
        f'<div class="quiet-card"><b>{profile.title}</b><br>'
        f'<small>代码：{profile.ticker} · 类型：{profile.kind_label}</small><br>'
        f'<small>{profile.category_label}：{profile.category_value}</small><br>'
        f'<small>{profile.subcategory_label}：{profile.subcategory_value}</small><br><br>'
        f'{profile.description}{weight_line}</div>',
        unsafe_allow_html=True,
    )


def overview(result: AnalyticsResult) -> None:
    metrics = [
        ("投资组合评分（Portfolio Score）", f"{result.portfolio_score:.0f}/100"),
        ("投资组合市值（Portfolio Value）", f"{result.snapshot.account_currency} {result.snapshot.total_assets:,.0f}"),
        ("风险等级（Risk Level）", RISK_LEVEL_LABELS.get(result.risk_level, result.risk_level)),
        ("前三大持仓占比（Top 3 Holdings）", f"{result.top3_concentration:.1%}"),
    ]
    items = ''.join(f'<div class="metric-item"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>' for label, value in metrics)
    st.markdown('<div class="metric-grid">' + items + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">直接持仓（Direct Holdings）</div>', unsafe_allow_html=True)
    rows = treemap_rows(result)
    frame = pd.DataFrame(rows)
    frame["label"] = frame.apply(lambda r: f"{r.ticker}<br>{r.weight:.1%}" + (f"<br>{r.daily_change:+.1%}" if pd.notna(r.daily_change) else ""), axis=1)
    fig = px.treemap(frame, path=["ticker"], values="weight", custom_data=["label"], color="weight", color_continuous_scale=list(reversed(BLUE)))
    fig.update_traces(texttemplate="%{customdata[0]}", hovertemplate="%{label}: %{value:.1%}<extra></extra>", marker_line_width=2, marker_line_color="#F7F9FC")
    fig.update_layout(height=430, margin=dict(l=0,r=0,t=8,b=0), coloraxis_showscale=False, paper_bgcolor="#F7F9FC")
    st.session_state.setdefault("treemap_selected", None)
    st.session_state.setdefault("treemap_nonce", 0)
    event = st.plotly_chart(
        fig, width="stretch", config={"displayModeBar": False},
        key=f"chart_treemap_{st.session_state.treemap_nonce}",
        on_select="rerun", selection_mode="points",
    )
    points = (event.get("selection", {}) or {}).get("points", []) if event else []
    clicked = points[0].get("label") if points else None
    if clicked and clicked != st.session_state.treemap_selected:
        # Plotly's treemap click-to-zoom otherwise persists across
        # Streamlit reruns because the frontend keeps its view state tied
        # to the component's key. Remounting under a fresh key forces a
        # clean, un-zoomed redraw on the very next run; the actual
        # selection survives the remount via session_state, not the
        # now-discarded widget's own event.
        st.session_state.treemap_selected = clicked
        st.session_state.treemap_nonce += 1
        st.rerun()
    selected = st.session_state.treemap_selected
    if selected and selected != "Other":
        # Layer 1, deterministic only: a local reference lookup
        # (core/security_profile.py). No network or LLM call.
        weight = next((r["weight"] for r in rows if r["ticker"] == selected), None)
        _security_profile_card(selected, weight)
    st.markdown('<div class="section-label">资产配置（Asset Allocation）</div>', unsafe_allow_html=True)
    alloc = pd.DataFrame([{"asset_key": k, "weight": v} for k, v in result.asset_allocation.items() if v > 0]).sort_values("weight", ascending=False)
    alloc["label"] = alloc.apply(lambda r: f"{ASSET_CLASS_LABELS_ZH.get(r.asset_key, r.asset_key)}（{ASSET_CLASS_LABELS_EN.get(r.asset_key, r.asset_key)}） {r.weight:.1%}", axis=1)
    donut = px.pie(alloc, names="label", values="weight", hole=.64, color="asset_key", color_discrete_map=DONUT_COLORS)
    donut.update_traces(textinfo="none", hovertemplate="%{label}<extra></extra>", marker=dict(line=dict(color="#F7F9FC", width=2)))
    donut.update_layout(
        height=280, margin=dict(l=0, r=0, t=8, b=0), showlegend=True, legend_title=None,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        paper_bgcolor="#F7F9FC",
        annotations=[dict(text="资产配置<br>100%", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#102A43"))],
    )
    st.plotly_chart(donut, width="stretch", config={"displayModeBar": False}, key="chart_asset_allocation")
    st.markdown(f'<div class="quiet-card">{result.summary}</div>', unsafe_allow_html=True)
    page_disclaimer()


def _rank_bars(rows: list[tuple[str, float]]) -> str:
    if not rows:
        return ""
    max_weight = max(w for _, w in rows) or 1
    items = "".join(
        f'<div class="rank-row"><div class="rank-num">{i}</div><div class="rank-ticker">{ticker}</div>'
        f'<div class="rank-track"><div class="rank-fill" style="width:{max(4.0, weight / max_weight * 100):.1f}%"></div></div>'
        f'<div class="rank-pct">{weight:.1%}</div></div>'
        for i, (ticker, weight) in enumerate(rows, start=1)
    )
    return f'<div class="rank-list">{items}</div>'


def _lookthrough_bars(exposures: list) -> str:
    if not exposures:
        return ""
    max_total = max(x.true for x in exposures) or 1
    legend = (
        '<div class="etf-legend"><span><i style="background:#24405C"></i>直接持仓（Direct）</span>'
        '<span><i style="background:#9FB8C9"></i>经ETF穿透（Via ETFs）</span></div>'
    )
    rows = "".join(
        f'<div class="etf-row"><div class="etf-ticker">{x.ticker}</div>'
        f'<div class="etf-track"><div class="etf-direct" style="width:{x.direct / max_total * 100:.1f}%"></div>'
        f'<div class="etf-indirect" style="width:{x.indirect / max_total * 100:.1f}%"></div></div>'
        f'<div class="etf-total">{x.true:.1%}</div></div>'
        for x in exposures
    )
    return legend + f'<div class="etf-list">{rows}</div>'


def risk_page(result: AnalyticsResult) -> None:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown('<div class="section-label">直接持仓集中度（Direct Concentration）</div>', unsafe_allow_html=True)
        st.markdown(_rank_bars(list(result.top_direct[:6])), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-label">ETF 穿透分析（ETF Look-through）</div>', unsafe_allow_html=True)
        exposure = [x for x in result.true_exposures if x.indirect > 0][:6]
        if exposure:
            st.markdown(_lookthrough_bars(exposure), unsafe_allow_html=True)
            if result.uncovered_etfs:
                # Partial coverage is not the same as no coverage: some ETFs
                # in the portfolio have verified look-through data and are
                # shown above; others don't and are fail-closed-excluded,
                # never guessed.
                st.caption("部分 ETF 暂无可验证底层持仓数据，穿透结果仅基于已验证部分。")
        else:
            st.info("当前组合暂无可验证的股票型 ETF 穿透数据。")
    st.markdown('<div class="section-label">穿透后真实暴露（True Exposure）</div>', unsafe_allow_html=True)
    top_true = list(result.true_exposures[:5])
    for offset in range(0, len(top_true), 2):
        cols = st.columns(2)
        for col, item in zip(cols, top_true[offset:offset + 2]):
            risk_class = "risk-number" if item.true > .15 else ""
            col.markdown(f'<div class="quiet-card"><b>{item.ticker}</b><br><span class="{risk_class}" style="font-size:1.35rem">{item.true:.1%}</span><br><small>{item.direct:.1%} 直接持仓 · {item.indirect:.1%} 经ETF穿透</small></div>', unsafe_allow_html=True)
    if result.risk_flags:
        st.markdown('<div class="section-label">风险提示（Risk Flags）</div>', unsafe_allow_html=True)
        for flag in result.risk_flags:
            badge = "risk-badge" if flag.severity == "high" else ""
            st.markdown(f'<div class="quiet-card" style="margin:.55rem 0"><span class="{badge}">{flag.title}</span><br><small>{flag.detail}</small></div>', unsafe_allow_html=True)
    page_disclaimer()


ROLE_LABELS = {
    "macro": "宏观与市场（Macro & Market）", "portfolio": "组合结构（Portfolio Structure）", "risk": "风险（Risk）",
    "valuation_data": "估值与数据（Valuation & Data）", "tax": "税务与账户（Tax & Account）",
    "action_rebalancing": "行动与再平衡（Action & Rebalancing）",
}


def committee_view(
    ai_result, *, holdings=None, quotes=None, cash=None, account_currency=None,
    usd_cad=None, account_type=None, before_result=None,
) -> None:
    """The trade_preview_context kwargs (all optional) are the same canonical
    holdings/quotes/snapshot inputs already used for the real analysis --
    passing them enables the Step 2A deterministic Trade Impact Preview
    under each REDUCE-type security action; omitting any of them simply
    disables that preview (e.g. in tests that don't need it), never changes
    the rest of the committee rendering."""
    st.markdown('<div class="section-label">七人投资委员会（Investment Committee）</div>', unsafe_allow_html=True)
    for offset in range(0, len(ai_result.members), 2):
        cols = st.columns(2)
        for col, member in zip(cols, ai_result.members[offset:offset + 2]):
            col.markdown(f'<div class="quiet-card member"><div class="member-role">{ROLE_LABELS[member.role]}</div><div class="member-stance">{member.stance}</div><div class="member-copy">{member.conclusion}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="quiet-card" style="margin-top:1rem"><b>多数意见（Majority View）</b><br>{ai_result.majority_view}<br><br><b>主要关切（Main Concern）</b><br>{ai_result.main_concern}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">主席决策（Chairman Decision）</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="quiet-card"><b>{ai_result.chairman_decision}</b></div>', unsafe_allow_html=True)
    _render_action_plan(
        ai_result.action_plan, holdings=holdings, quotes=quotes, cash=cash,
        account_currency=account_currency, usd_cad=usd_cad, account_type=account_type,
        before_result=before_result,
    )


_PRIORITY_CSS = {"高": "ap-priority-high", "中高": "ap-priority-medhigh", "中": "ap-priority-med", "低": "ap-priority-low"}
_ACTION_LABELS_ZH = {
    "HOLD": "持有", "WAIT": "观望", "REDUCE": "减仓", "REDUCE_ON_REBOUND": "反弹后减仓",
    "ADD_ON_PULLBACK": "回调后加仓", "STAGED_BUY": "分批买入", "STAGED_SELL": "分批卖出",
    "CONTROL_ADDITIONS": "控制新增", "REVIEW_AFTER_EVENT": "事件后复核",
}
_TIMELINE_LABELS = [
    ("timeline_now", "现在（Now）"), ("timeline_30_days", "未来30天（Next 30 Days）"),
    ("timeline_3_months", "未来3个月（Next 3 Months）"), ("timeline_6_12_months", "未来6–12个月（6–12 Months）"),
]


def _format_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def _trade_impact_preview_html(action, *, holdings, quotes, cash, account_currency, usd_cad, account_type, before_result) -> str:
    """Returns a compact HTML snippet to nest inside the action's own
    security card, or "" when no preview should be shown (missing context,
    non-REDUCE action, or any Step 2A suppression rule -- see
    core.trade_preview.build_trade_impact_preview)."""
    if None in (holdings, quotes, cash, account_currency, account_type, before_result):
        return ""
    preview = build_trade_impact_preview(
        holdings=holdings, quotes=quotes, cash=cash, account_currency=account_currency,
        usd_cad=usd_cad, account_type=account_type, before_result=before_result,
        ticker=action.ticker, action=action.action, position_reduction_pct=action.position_reduction_pct,
    )
    if preview is None:
        return ""
    # An identified-exposure figure only needs the Step 1 lower-bound "≥"
    # framing when this ticker actually has an indirect (ETF-sourced)
    # component -- a purely direct holding's true exposure is already exact.
    has_indirect = (
        preview.before_identified_exposure is not None
        and preview.before_direct_weight is not None
        and preview.before_identified_exposure > preview.before_direct_weight + 1e-9
    )
    exposure_prefix = "≥" if has_indirect else ""
    gain_loss_line = (
        f"约 {preview.trade_currency} {preview.estimated_realized_gain_loss:,.0f}（基于已保存的平均成本，估算值，非税务估算）"
        if preview.gain_loss_available else "暂不可估算（平均成本币种未确认）"
    )
    return (
        f'<div class="ap-security-meta" style="margin-top:.5rem;padding-top:.5rem;border-top:1px dashed #E2EAF4">'
        f'<b>交易影响预览（Trade Impact Preview）</b><br>'
        f'请求：当前仓位减少约 {preview.requested_position_reduction_pct:.0f}%'
        f'（可执行约 {preview.executable_shares:.0f} 股，对应约 {preview.executable_position_reduction_pct:.1f}%）<br>'
        f'预计交易金额：约 {preview.trade_currency} {preview.trade_value:,.0f}<br>'
        f'预计已实现盈亏：{gain_loss_line}<br>'
        f'直接权重：{_format_pct(preview.before_direct_weight)} → {_format_pct(preview.after_direct_weight)}　'
        f'已识别穿透暴露：{exposure_prefix}{_format_pct(preview.before_identified_exposure)} → '
        f'{exposure_prefix}{_format_pct(preview.after_identified_exposure)}　'
        f'Top-5 集中度：{_format_pct(preview.before_top5_concentration)} → {_format_pct(preview.after_top5_concentration)}<br>'
        f'<span style="font-size:.75rem;color:#8AA0B8">假设：卖出所得暂存为现金，不预设再投资标的；本预览为本地确定性估算，不构成税务建议。</span>'
        f'</div>'
    )


def _render_action_plan(
    plan, *, holdings=None, quotes=None, cash=None, account_currency=None,
    usd_cad=None, account_type=None, before_result=None,
) -> None:
    st.markdown('<div class="section-label">条件式行动方案（Action Plan）</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">当前行动结论（Current Action Summary）</div>', unsafe_allow_html=True)
    top_actions = "".join(f"<li>{c}</li>" for c in plan.top_actions)
    do_now = "".join(f"<li>{d}</li>" for d in plan.do_now)
    do_not = "".join(f"<li>{d}</li>" for d in plan.do_not_now)
    st.markdown(
        f'<div class="quiet-card"><b>当前总策略（Strategy Now）</b><br>{plan.strategy_now}'
        f'<div style="margin-top:.7rem"><b>本阶段最重要的几件事（Top Actions）</b>'
        f'<ol style="margin:.35rem 0 0;padding-left:1.2rem">{top_actions}</ol></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ap-donow-grid">'
        f'<div class="quiet-card"><b>现在做（Do Now）</b><ul class="ap-list ap-do-now">{do_now}</ul></div>'
        f'<div class="quiet-card"><b>现在不做（Do Not Do Now）</b><ul class="ap-list ap-do-not">{do_not}</ul></div>'
        f'</div>', unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">重点持仓行动（Key Security Actions）</div>', unsafe_allow_html=True)
    for action in plan.security_actions:
        action_zh = _ACTION_LABELS_ZH.get(action.action, action.action)
        meta = (
            f"<b>目标</b> {action.target}<br><b>触发条件</b> {action.trigger}<br><b>理由</b> {action.reason}"
        )
        priority_css = _PRIORITY_CSS.get(action.priority, "ap-priority-med")
        trade_preview_html = _trade_impact_preview_html(
            action, holdings=holdings, quotes=quotes, cash=cash, account_currency=account_currency,
            usd_cad=usd_cad, account_type=account_type, before_result=before_result,
        )
        st.markdown(
            f'<div class="ap-security-card">'
            f'<div class="ap-security-head"><span class="ap-security-ticker">{action.ticker}</span>'
            f'<span class="ap-priority {priority_css}">优先级 {action.priority}</span></div>'
            f'<div class="ap-security-action">{action_zh}</div>'
            f'<div class="ap-security-meta">{meta}</div>'
            f'{trade_preview_html}'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">行动时间线（Action Timeline）</div>', unsafe_allow_html=True)
    horizon_cards = ""
    for key, label in _TIMELINE_LABELS:
        actions_html = "".join(f"<li>{a}</li>" for a in getattr(plan, key))
        horizon_cards += (
            f'<div class="ap-timeline-card"><div class="ap-timeline-title">{label}</div>'
            f'<ul class="ap-list">{actions_html}</ul></div>'
        )
    st.markdown(f'<div class="ap-timeline-grid">{horizon_cards}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">关键事件与重新评估条件（Key Events &amp; Reassessment Triggers）</div>', unsafe_allow_html=True)
    trigger_rows = "".join(
        f'<div class="ap-calendar-row" style="grid-template-columns:1fr 1fr">'
        f'<div><span class="ap-calendar-event">{trig.event_or_condition}</span>'
        + (f'<div class="ap-calendar-reassess">关注 {" · ".join(trig.affected_holdings)}</div>' if trig.affected_holdings else "")
        + '</div>'
        f'<span class="ap-calendar-reassess">{trig.reassess}</span></div>'
        for trig in plan.reassessment_triggers
    )
    st.markdown(f'<div class="quiet-card">{trigger_rows}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">执行清单（Execution Checklist）</div>', unsafe_allow_html=True)
    checklist_items = "".join(f"<li>{c}</li>" for c in plan.checklist)
    st.markdown(f'<div class="quiet-card"><ul class="ap-checklist">{checklist_items}</ul></div>', unsafe_allow_html=True)
