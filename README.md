# 加美通 AI 投资分析
CanAm AI Investment Analytics

**Status: Public Beta / Final MVP frozen** (see [docs/FINAL_MVP_RELEASE.md](docs/FINAL_MVP_RELEASE.md))

A three-page portfolio analysis MVP for Canadian and U.S. stocks and ETFs.

- **Public app:** https://canam-ai-investment.streamlit.app
- **Repository:** https://github.com/skylandvan-pixel/CanAm-AI-Investment-Analytics

## What it does

Three pages:

1. **投资组合概览** — Portfolio Overview: Portfolio Score, Portfolio Value, Risk Level, Direct Holdings Treemap (click a holding for its Security Profile), Asset Allocation.
2. **真实风险与 ETF 穿透** — Risk & ETF Look-through: Direct Concentration, ETF Look-through, True Exposure, deterministic risk flags.
3. **AI 投资委员会** — AI Investment Committee: a seven-member AI investment committee, chairman decision, and a conditional action plan (Free vs Pro).

## Core Analytics vs AI Intelligence

**Layer 1 — Core Analytics** (deterministic, no LLM required): holdings, weights, allocation, concentration, ETF look-through, direct/indirect/true exposure, Effective N, Portfolio Score, risk flags. Fully functional with zero AI provider configured.

**Layer 2 — AI Intelligence** (optional, Pro/Beta only): a seven-member AI investment committee that reads the canonical Layer 1 fact packet and returns majority view, chairman decision, and a conditional action plan. It never invents or overrides deterministic financial values (holdings, weights, prices, scores, exposure). A Layer 2 failure never breaks Layer 1.

## Key Features

- Deterministic Layer 1 analytics that work with zero AI configuration.
- CSV/XLSX portfolio import into a single canonical holdings table.
- Canadian/U.S. ticker identity resolution and quote-metadata validation (fail-closed on mismatch or ambiguity — never a guessed security).
- ETF look-through with verified-only holdings data; partial coverage is reported, never faked.
- Optional Layer 2: a seven-member AI investment committee, explicit-trigger only — the AI is called only when a Pro/Beta-unlocked user clicks "启动 AI 投资委员会"; page navigation, reruns, and Beta Code unlock never call the AI provider.
- AI provider (Gemini) uses bounded automatic retry with backoff on transient 5xx/high-demand errors; permanent errors (invalid request, auth) fail closed immediately, never retried.

## Free vs Pro

**Free — $0**
Portfolio Overview, Portfolio Score, Treemap, Asset Allocation, Direct Concentration, ETF Look-through.

**Pro — US$19.90/month** (displayed MVP price)
Everything in Free, plus the seven-member AI investment committee, chairman decision, personalized action plan, and staged (now / 1 month / 3 month) investment plan.

Payment is not yet enabled during Public Beta. The "升级 Pro" button only shows an informational message — no checkout, no billing. Beta users can access Pro features with an invitation Beta Code.

## Tech Stack

- Python, [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/python/)
- [yfinance](https://github.com/ranaroussi/yfinance) for market quotes and FX
- [Pydantic](https://docs.pydantic.dev/) for AI response schema validation
- AI providers (Layer 2, optional): [Google Gemini](https://ai.google.dev/) (`google-genai`, default provider, production model `gemini-3.6-flash`) or [Anthropic Claude](https://www.anthropic.com/) (`anthropic`, configurable alternative)

See `requirements.txt` for exact version constraints.

## Repository Structure

```
app.py                  Streamlit entry point / page routing
ui.py                   shared UI rendering helpers
core/analytics.py       deterministic Layer 1 portfolio analytics
core/market.py          market data / FX (yfinance)
core/ticker_resolution.py  Canadian/U.S. ticker identity resolution
core/reference.py       verified ETF look-through holdings data
core/security_profile.py  Security Profile lookup
core/importers.py       CSV/XLSX portfolio import
core/auth.py            Beta Code validation
core/ai.py              Layer 2: Gemini/Anthropic providers, Seven-Member Committee
core/demo.py            demo portfolio state
core/models.py          shared data models
tests/                  pytest suite
docs/                   canonical MVP spec + release records
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in only what you need — Layer 1 works with none of these set.

```
AI_PROVIDER=gemini
GEMINI_API_KEY=YOUR_API_KEY
ANTHROPIC_API_KEY=YOUR_API_KEY
CANAM_BETA_CODES=
```

Never commit `.env`.

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

## Security & Privacy

- Secrets are not stored in Git. Local development uses environment variables (`.env`, gitignored); Streamlit Cloud deployments use Streamlit Secrets.
- Diagnostic logs are sanitized server-side (API keys, Beta Codes, and known secret/token shapes are redacted) and are never shown in the UI.
- A portfolio's fact packet is sent to the configured AI provider only after explicit user action (unlocking Pro and clicking "启动 AI 投资委员会") — never on navigation, page load, or rerun.
- There is no user account system and no database — holdings are processed only in the current browser session.

## Disclaimer

免责声明：本系统内容由 AI 与数据模型生成，仅供信息与分析参考，不构成任何投资建议。

Disclaimer: AI- and data-generated content is for informational and analytical purposes only and does not constitute investment advice.

This product has no payment backend, no brokerage/trade execution, and is not a registered investment advisor.

## Release / Version Reference

Latest release: `canam-mvp-v1.0.4` (and the frozen baseline tag `canam-mvp-final`). Full lineage and status in [docs/FINAL_MVP_RELEASE.md](docs/FINAL_MVP_RELEASE.md).

## Documentation

- [docs/CanAm_AI_Investment_Analytics_MVP.md](docs/CanAm_AI_Investment_Analytics_MVP.md) — canonical MVP product/technical specification
- [docs/FINAL_MVP_RELEASE.md](docs/FINAL_MVP_RELEASE.md) — final release record and freeze declaration
