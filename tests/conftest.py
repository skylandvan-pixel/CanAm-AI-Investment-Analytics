from __future__ import annotations

import pytest

from core.analytics import analyze, build_snapshot
from core.models import HoldingInput, Quote


@pytest.fixture(autouse=True)
def _no_live_market_regime_network_by_default(monkeypatch):
    """Step 2A.8: ui.overview() fetches live SPY/VIX history for the Market
    Regime module on every Page 1 render. Without this guard, every
    AppTest-based test in tests/test_app.py would make a real yfinance
    network call, making the whole suite slow and network-dependent (the
    project's existing tests never depend on live network -- AI providers
    and quotes are always injected/mocked). Defaults every test to a fast,
    deterministic "unavailable" snapshot (itself exercising the real
    fail-closed/omit-when-unavailable path); a test that needs specific
    Market Regime content overrides this with its own
    monkeypatch.setattr(ui, "get_market_regime_snapshot", ...), which wins
    since it runs after this fixture within the same test."""
    import ui
    from core.market_regime import MarketRegimeSnapshot

    monkeypatch.setattr(ui, "get_market_regime_snapshot", lambda *a, **k: MarketRegimeSnapshot())


@pytest.fixture
def quote_factory():
    def make(*tickers):
        return {ticker: Quote(ticker, 100.0, "USD", 99.0, "test", "verified") for ticker in tickers}
    return make


@pytest.fixture
def mixed_result(quote_factory):
    holdings = [HoldingInput("NVDA", 20), HoldingInput("VOO", 50), HoldingInput("SGOV", 30)]
    return analyze(build_snapshot(holdings, quote_factory("NVDA", "VOO", "SGOV")))


@pytest.fixture
def low_confidence_result(quote_factory):
    """A portfolio with one materially incomplete look-through ETF (XLF has
    no published constituent data -- see core.reference.ETF_HOLDINGS) held
    alongside otherwise non-concentrated direct holdings, so no deterministic
    risk_flags trigger (risk_level == "Measured") even though look-through
    coverage is "insufficient" and a large share of the portfolio's true
    exposure cannot be verified. Used by the Step 2A.3 P0-1/P0-2
    risk-confidence-language guardrail tests (core.ai.
    _validate_risk_confidence_language)."""
    holdings = [
        HoldingInput("NVDA", 10), HoldingInput("AAPL", 10),
        HoldingInput("JPM", 10), HoldingInput("XOM", 10),
        HoldingInput("CBIL", 30), HoldingInput("XLF", 40),
    ]
    tickers = ("NVDA", "AAPL", "JPM", "XOM", "CBIL", "XLF")
    return analyze(build_snapshot(holdings, quote_factory(*tickers), cash=50))
