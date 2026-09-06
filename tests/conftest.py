from __future__ import annotations

import pytest

from core.analytics import analyze, build_snapshot
from core.models import HoldingInput, Quote


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
