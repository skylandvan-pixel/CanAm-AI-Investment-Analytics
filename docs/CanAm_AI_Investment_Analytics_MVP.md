# 加美通 AI 投资分析
## CanAm AI Investment Analytics
### MVP 产品与技术说明

This is the canonical project document for future development. It describes only what exists in the codebase today, not planned or aspirational features.

---

## A. Product Positioning

加美通 AI 投资分析（CanAm AI Investment Analytics）是一套面向加拿大和美国股票 / ETF 投资者的投资组合分析工具。

核心目标：让普通投资者用约 2 分钟回答三个问题。

- **Page 1** — 我持有什么？（What do I own?）
- **Page 2** — 我的真实风险在哪里？（Where is my real risk?）
- **Page 3** — AI 如何帮助我理解下一步行动？（What should I pay attention to and how might I act?）

明确定位：

- 不是券商（not a brokerage）
- 不是交易执行平台（not a trade-execution platform）
- 不是注册投资顾问（not a registered investment advisor）
- 不保证收益（no return is guaranteed）

---

## B. Three-Page MVP

### Page 1 — 投资组合概览（Portfolio Overview）

- Portfolio Score
- Portfolio Value
- Risk Level（Chinese-only value: 高 / 偏高 / 适中）
- Top 3 Holdings
- Direct Holdings Treemap
- Click a treemap holding → Security Profile renders directly below (no dropdown, no persistent zoom)
- Asset Allocation donut — 股票（Equity）/ 债券与现金类（Bonds & Cash-like）/ 现金（Cash）
- Concise deterministic one-line summary

### Page 2 — 真实风险与 ETF 穿透（Risk & ETF Look-through）

- Direct Concentration (thin ranked bars)
- ETF Look-through (thin segmented bars: Direct = dark blue, Via ETFs = light blue)
- True Exposure cards
- Verified partial coverage messaging
- Deterministic risk flags

### Page 3 — AI 投资委员会（AI Investment Committee）

Free vs Pro membership presentation (see sections D/E below).

---

## C. Free

**$0 · 永久免费**

当前功能：

- 投资组合概览
- Portfolio Score
- Treemap 持仓分析
- 资产配置分析
- 持仓集中度分析
- ETF 穿透与真实风险

---

## D. Pro

**Displayed MVP pricing: US$19.90 / 月**

功能：

- 包含 Free 全部功能
- 七人 AI 投资委员会
- 主席综合决议
- 个性化行动方案
- 阶段性投资计划
- 深度 AI 组合解读

**IMPORTANT:** US$19.90 / 月 is an MVP pricing presentation / validation figure. There is currently no real billing or payment backend. The "升级 Pro" button never performs a payment — it only shows an informational message. Beta users access Pro via an invitation Beta Code.

---

## E. Two-Layer Architecture

**Layer 1 — Core Analytics** (`core/analytics.py`, `core/market.py`, `core/models.py`, `core/ticker_resolution.py`, `core/reference.py`, `core/security_profile.py`)

Fully deterministic. Does not require an LLM to run. Covers valuation, allocation, concentration, ETF look-through, direct/indirect/true exposure, portfolio score, risk flags, and the Security Profile lookup.

**Layer 2 — AI Intelligence** (`core/ai.py`)

- Seven-Member Investment Committee (six specialists + chairman)
- Chairman Decision
- Action Plan (Now / 1 month / 3 month conditional planning)
- Natural-language synthesis

**Principle:** AI consumes the canonical Layer 1 fact packet (`canonical_fact_packet`). AI does not independently invent or override deterministic financial facts — it cannot recalculate portfolio value, score, risk level, weights, true exposure, ETF look-through, or FX.

---

## F. Seven-Member Committee

Implemented in `core/ai.py::CommitteeResult` — six specialist roles, each contributing one conclusion, plus a chairman synthesis:

- macro（宏观与市场）
- portfolio（组合结构）
- risk（风险）
- valuation_data（估值与数据）
- tax（税务与账户）
- action_rebalancing（行动与再平衡）

6 specialists + Chairman = 7. This section describes only the architecture that exists in code today.

---

## G. ETF Look-through

```
True Exposure = Direct Exposure + Verified ETF Indirect Exposure
```

Principles:

- Verified holdings only — an ETF's Top-N constituent weights must be present in `core/reference.py::ETF_HOLDINGS`.
- Partial coverage ≠ zero coverage — if some but not all ETFs in a portfolio have verified holdings, the verified results are shown, with a caption noting partial coverage.
- Unknown ETF holdings → fail closed. Constituents are never guessed or invented.

**Current MVP coverage limitation:** ZSP currently has verified look-through coverage (it replicates the same S&P 500 index already verified for VOO/SPY/IVV). FINN, VDY, and VHT do not currently have verified Top-N holdings data in this codebase and are reported as uncovered — this is a data-coverage limitation, not a defect, and must not be worked around by inventing holdings.

---

## H. Security Identity

One ticker maps to exactly one canonical security identity, shared by market-data fetching (`core/market.py`), ticker resolution (`core/ticker_resolution.py`), and the Security Profile feature (`core/security_profile.py`).

**FINN** is the flagship example of why this matters: the bare ticker "FINN" collides with an unrelated U.S. OTC equity (First National of Nebraska, Inc.). The resolved identity must always be:

```
FINN → FINN.NE → Fidelity Global Innovators ETF Series L
```

never the unrelated U.S. OTC security.

Current verified mapping examples (`core/ticker_resolution.py::CANADIAN_REFERENCE`):

| Display ticker | Resolved market ticker |
|---|---|
| CBIL | CBIL.TO |
| ZSP | ZSP.TO |
| VDY | VDY.TO |
| XSB | XSB.TO |
| FINN | FINN.NE |

This is a set of **verified mapping examples**, not a universal rule for all Canadian securities. A ticker outside this reference is treated as a standard (typically U.S.) equity and validated against whatever metadata its quote actually returns — it is never assumed to be Canadian.

---

## I. Fail-Closed

The following must never produce a fabricated result:

- Unknown security
- Unknown ETF holdings
- Bad quote identity (currency/exchange/quote-type mismatch)
- Missing market data
- AI provider failure
- Missing secret (API key / Beta code not configured)
- Invalid Beta Code

A Layer 2 (AI) failure must never break Layer 1. Layer 1 must remain fully functional with zero AI provider configured.

---

## J. Providers

- **Market / reference data:** `yfinance` (Yahoo Finance) — used for live quotes and USD/CAD FX. This is an active dependency.
- **AI (Layer 2, optional):** Gemini (`google-genai`) is the default provider (`AI_PROVIDER=gemini`), production model `gemini-3.6-flash`; Anthropic (`anthropic`) is a configurable alternative (`AI_PROVIDER=anthropic`). Neither is called unless a user has unlocked Pro and explicitly clicks "启动 AI 投资委员会" — page navigation and reruns never trigger an API call. Gemini calls retry automatically (bounded, with backoff) on transient 5xx/high-demand errors; permanent errors (4xx: invalid request, auth) fail closed immediately without retry.

---

## K. Security / Secrets

- API keys must never enter Git.
- **Local development:** environment variables via `.env` (see `.env.example`; never commit `.env`).
- **Streamlit Cloud:** Streamlit Secrets (configured in the Cloud UI, not in the repository).
- A secret audit must be performed before any public deployment or public repository push.

---

## L. Disclaimer

Canonical disclaimer, shown identically at the bottom of all three pages:

> 免责声明：本系统内容由 AI 与数据模型生成，仅供信息与分析参考，不构成任何投资建议。
>
> Disclaimer: AI- and data-generated content is for informational and analytical purposes only and does not constitute investment advice.

---

## M. Testing Baseline

Release baseline (MVP v1.0):

```
122 passed
1 skipped
0 failed
```

Current baseline (canam-mvp-v1.0.4 / canam-mvp-final):

```
147 passed
1 skipped
0 failed
```

Also verified:

- Desktop browser acceptance passed
- 375px mobile acceptance passed — no horizontal overflow
- No duplicate Analyze button
- No `removeChild` errors
- No `NotFoundError` errors

---

## N. Known MVP Limitations

- ETF holdings coverage is partial (see section G) — FINN, VDY, VHT currently lack verified look-through data.
- No production payment backend.
- No production subscription management.
- Pro access during Beta is invitation-code only (`CANAM_BETA_CODES`).
- External market/reference data availability depends on the `yfinance`/Yahoo Finance provider.
- Gemini may experience temporary 5xx/high-demand errors; bounded automatic retry absorbs most transient cases, but Layer 2 still depends on external provider availability.
- Public Beta may reveal additional edge cases not covered by the current test suite.

---

## O. FINAL MVP STATUS

- **MVP frozen date:** 2026-09-03
- **Production URL:** https://canam-ai-investment.streamlit.app
- **Repository:** https://github.com/skylandvan-pixel/CanAm-AI-Investment-Analytics
- **Latest validated release:** `canam-mvp-v1.0.4`, frozen baseline tag `canam-mvp-final`
- **Production Layer 1:** PASS
- **Production Layer 2:** PASS
- **Final tests:** 147 passed, 1 skipped, 0 failed
- **Feature freeze status:** FROZEN — see [docs/FINAL_MVP_RELEASE.md](FINAL_MVP_RELEASE.md) for the freeze declaration and change-classification rules

**Known non-blocking limitations** (see section N for the full list): partial third-party ETF holdings coverage; Gemini may experience temporary 5xx high-demand errors (mitigated by bounded retry); AI depends on external provider availability; Pro is Beta-code gated, not a paid subscription; no account persistence/database; no trade execution. None of these are release blockers.
