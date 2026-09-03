# Read-only legacy analytics audit

Audited source: private legacy repository `50-US-Stock-ETF-Investment-System` (local checkout, not part of this repository)

Verified baseline:

- Branch: `main`
- HEAD and `origin/main`: `6308824bd165174a89f1928ceb0bf7d9bad49c14`
- Commit: `fix: P0.15-Lite presentation consistency cleanup`
- Existing untracked Markdown was observed and left untouched.

## Adapted deterministic logic

| Legacy source | MVP destination | Preserved contract |
|---|---|---|
| `scripts/valuation.py` | `core/analytics.py::build_snapshot` | One valuation path owns value and weight; missing price/FX fails closed |
| `scripts/portfolio_analytics.py` | `core/analytics.py::analyze` | Allocation, concentration, liquidity reserve, SGOV/CBIL classification |
| `scripts/overlap.py` | `core/analytics.py::analyze` | Top-N reference only; direct + ETF contribution = true exposure; no guessing |
| `scripts/portfolio_diversification.py` | `core/analytics.py::_effective_n` | Direct and look-through Effective N remain distinct |
| `scripts/scoring.py` | `core/analytics.py::_score` | Six structural components renormalized to 100; data coverage not blended into quality |
| `scripts/market_data.py` | `core/market.py` | Previous-close provider, Canadian symbol identity guard, provider failure closes data path |
| `scripts/ai_provider.py` | `core/ai.py` | Explicit provider selection, Gemini/Anthropic adapters, no automatic fallback |
| `scripts/advanced_committee.py` | `core/ai.py` | Six canonical specialist roles plus chairman; structured result and conditional actions |

Not migrated: legacy report composition, monitoring UI, tax/rates/oil/options/geopolitical modules, diagnostics, and long-form commentary. They do not belong in the three-screen MVP.

## Canonical content map

- Direct allocation → Page 1 treemap
- Asset allocation and Portfolio Score → Page 1
- Direct concentration, ETF look-through, true exposures and risk flags → Page 2
- Committee, chairman decision and Now / 1M / 3M plan → Page 3
