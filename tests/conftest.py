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
