# 加美通 AI 投资分析
CanAm AI Investment Analytics

A three-page portfolio analysis MVP for Canadian and U.S. stocks and ETFs.

## What it does

Three pages:

1. **投资组合概览** — Portfolio Overview: Portfolio Score, Portfolio Value, Risk Level, Direct Holdings Treemap (click a holding for its Security Profile), Asset Allocation.
2. **真实风险与 ETF 穿透** — Risk & ETF Look-through: Direct Concentration, ETF Look-through, True Exposure, deterministic risk flags.
3. **AI 投资分析与行动方案** — AI Analysis & Action Plan: a seven-member AI investment committee, chairman decision, and a conditional action plan (Free vs Pro).

## Key Features

- Deterministic Layer 1 analytics that work with zero AI configuration.
- CSV/XLSX portfolio import into a single canonical holdings table.
- Canadian/U.S. ticker identity resolution and quote-metadata validation (fail-closed on mismatch or ambiguity — never a guessed security).
- ETF look-through with verified-only holdings data; partial coverage is reported, never faked.
- Optional Layer 2: a seven-member AI investment committee, explicit-trigger only.

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
- Optional AI providers: [Google Gemini](https://ai.google.dev/) (`google-genai`) or [Anthropic Claude](https://www.anthropic.com/) (`anthropic`)

See `requirements.txt` for exact version constraints.

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

## Security

Secrets are not stored in Git. Local development uses environment variables (`.env`, gitignored); Streamlit Cloud deployments use Streamlit Secrets.

## Disclaimer

免责声明：本系统内容由 AI 与数据模型生成，仅供信息与分析参考，不构成任何投资建议。

Disclaimer: AI- and data-generated content is for informational and analytical purposes only and does not constitute investment advice.

## Documentation

[docs/CanAm_AI_Investment_Analytics_MVP.md](docs/CanAm_AI_Investment_Analytics_MVP.md)
