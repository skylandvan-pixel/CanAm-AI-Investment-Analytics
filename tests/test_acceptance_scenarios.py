from __future__ import annotations

import os

import pytest

from core.ai import ProviderUnavailable, configured_provider
from core.analytics import analyze, build_snapshot
from core.auth import validate_beta_code
from core.models import HoldingInput, Quote


def quotes(*tickers):
    return {ticker: Quote(ticker, 100, "USD", 99, "scenario", "verified") for ticker in tickers}


def test_scenario_a_stocks_only():
    result = analyze(build_snapshot([HoldingInput("AAPL", 6), HoldingInput("JPM", 4)], quotes("AAPL", "JPM")))
    assert result.asset_allocation == {"Equity": 1.0}
    assert result.lookthrough_coverage == "complete"


def test_scenario_b_etf_heavy():
    result = analyze(build_snapshot([HoldingInput("VOO", 6), HoldingInput("XLK", 4)], quotes("VOO", "XLK")))
    assert result.true_exposures
    assert all(item.indirect > 0 for item in result.true_exposures)


def test_scenario_c_mixed_hidden_overlap():
    result = analyze(build_snapshot([HoldingInput("NVDA", 2), HoldingInput("VOO", 8)], quotes("NVDA", "VOO")))
    nvda = next(item for item in result.true_exposures if item.ticker == "NVDA")
    assert nvda.true > nvda.direct


def test_scenario_d_fixed_income_sgov():
    result = analyze(build_snapshot([HoldingInput("SGOV", 7), HoldingInput("AAPL", 3)], quotes("SGOV", "AAPL")))
    assert result.asset_allocation["Fixed Income"] == pytest.approx(.7)
    assert "SGOV" not in result.uncovered_etfs


def test_scenario_e_missing_partial_data():
    missing = Quote("MISSING", None, None, status="error:test")
    snapshot = build_snapshot([HoldingInput("AAPL", 1), HoldingInput("MISSING", 1)], {"AAPL": quotes("AAPL")["AAPL"], "MISSING": missing})
    assert snapshot.coverage_ratio < 1
    assert snapshot.positions[1].status == "incomplete"


def test_scenario_f_ai_api_unavailable(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable):
        configured_provider()


def test_scenario_g_ai_locked():
    assert not validate_beta_code("wrong", ("BETA",))


@pytest.mark.skipif(not (os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")), reason="No safely configured real AI key")
def test_scenario_h_real_provider_is_configured_only_when_key_exists():
    assert configured_provider() is not None
