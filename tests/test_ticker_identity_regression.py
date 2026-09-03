"""P0 regression: CBIL/ZSP/FINN/VDY/XSB/NVDA/VHT in a CAD account must
never reproduce the ~CAD 96M / ~99.5% FINN bug caused by bare "FINN"
colliding with an unrelated U.S. OTC stock (First National of Nebraska).

All yfinance access is mocked with fixed, deterministic data captured
from real Yahoo quotes so this test never depends on live market prices.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.analytics import analyze, build_snapshot
from core.market import fetch_quotes
from core.models import HoldingInput

PORTFOLIO = [
    HoldingInput("CBIL", 3218),
    HoldingInput("ZSP", 1250),
    HoldingInput("FINN", 3500),
    HoldingInput("VDY", 900),
    HoldingInput("XSB", 2500),
    HoldingInput("NVDA", 150),
    HoldingInput("VHT", 100),
]

# (price, currency, quoteType) keyed by the exact symbol yfinance.Ticker
# should be constructed with. "FINN" (bare) is deliberately the real,
# unrelated U.S. OTC collision (First National of Nebraska, ~$19,750/share)
# to prove the resolver never queries it.
MARKET_DATA = {
    "CBIL.TO": (50.01, "CAD", "ETF"),
    "ZSP.TO": (116.34, "CAD", "ETF"),
    "FINN.NE": (31.90, "CAD", "ETF"),
    "VDY.TO": (77.32, "CAD", "ETF"),
    "XSB.TO": (26.66, "CAD", "ETF"),
    "NVDA": (224.41, "USD", "EQUITY"),
    "VHT": (324.47, "USD", "ETF"),
    "FINN": (19750.0, "USD", "EQUITY"),  # must never be queried
}


class _FakeTicker:
    queried_symbols: list[str] = []

    def __init__(self, symbol: str):
        self.symbol = symbol
        _FakeTicker.queried_symbols.append(symbol)

    def history(self, period="10d", interval="1d", auto_adjust=False):
        data = MARKET_DATA.get(self.symbol)
        if data is None:
            return pd.DataFrame()
        price = data[0]
        return pd.DataFrame({"Close": [price * 0.99, price]}, index=pd.date_range("2026-01-01", periods=2))

    @property
    def fast_info(self):
        data = MARKET_DATA.get(self.symbol)
        if data is None:
            return {}
        _, currency, quote_type = data
        return {"currency": currency, "quoteType": quote_type}


@pytest.fixture
def fake_yfinance(monkeypatch):
    import yfinance
    _FakeTicker.queried_symbols = []
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)
    return _FakeTicker


def test_finn_resolves_via_ne_suffix_never_bare(fake_yfinance):
    fetch_quotes(["FINN"])
    assert "FINN.NE" in fake_yfinance.queried_symbols
    assert "FINN" not in fake_yfinance.queried_symbols


def test_finn_quote_is_the_canadian_etf_not_the_us_collision(fake_yfinance):
    quotes = fetch_quotes(["FINN"])
    assert quotes["FINN"].status == "verified"
    assert quotes["FINN"].price == pytest.approx(31.90)
    assert quotes["FINN"].currency == "CAD"


def test_all_seven_tickers_resolve_with_correct_currency(fake_yfinance):
    quotes = fetch_quotes([h.ticker for h in PORTFOLIO])
    assert quotes["CBIL"].currency == "CAD"
    assert quotes["ZSP"].currency == "CAD"
    assert quotes["FINN"].currency == "CAD"
    assert quotes["VDY"].currency == "CAD"
    assert quotes["XSB"].currency == "CAD"
    assert quotes["NVDA"].currency == "USD"
    assert quotes["VHT"].currency == "USD"
    for ticker, quote in quotes.items():
        assert quote.status == "verified", f"{ticker} unexpectedly unresolved: {quote.status}"


def test_no_symbol_silently_maps_to_a_different_security(fake_yfinance):
    fetch_quotes([h.ticker for h in PORTFOLIO])
    # Every resolved-Canadian ticker must have queried its own mapped
    # symbol, never a bare/incorrect one that happens to also return data.
    assert "CBIL.TO" in fake_yfinance.queried_symbols and "CBIL" not in fake_yfinance.queried_symbols
    assert "ZSP.TO" in fake_yfinance.queried_symbols and "ZSP" not in fake_yfinance.queried_symbols
    assert "VDY.TO" in fake_yfinance.queried_symbols and "VDY" not in fake_yfinance.queried_symbols
    assert "XSB.TO" in fake_yfinance.queried_symbols and "XSB" not in fake_yfinance.queried_symbols
    assert "FINN.NE" in fake_yfinance.queried_symbols and "FINN" not in fake_yfinance.queried_symbols


def test_exact_portfolio_produces_plausible_value_and_weights(fake_yfinance):
    quotes = fetch_quotes([h.ticker for h in PORTFOLIO])
    snapshot = build_snapshot(PORTFOLIO, quotes, account_currency="CAD", usd_cad=1.36)
    result = analyze(snapshot)

    # Hand-computed from MARKET_DATA above: CBIL 3218*50.01 + ZSP 1250*116.34
    # + FINN 3500*31.90 + VDY 900*77.32 + XSB 2500*26.66 (all CAD) plus
    # NVDA 150*224.41 and VHT 100*324.47 (USD, *1.36 FX).
    expected_total = (
        3218 * 50.01 + 1250 * 116.34 + 3500 * 31.90 + 900 * 77.32 + 2500 * 26.66
        + (150 * 224.41 + 100 * 324.47) * 1.36
    )
    assert snapshot.total_assets == pytest.approx(expected_total, rel=1e-6)
    assert snapshot.total_assets < 1_000_000  # nowhere near the reported ~CAD 96,059,891

    weights = dict(result.top_direct)
    assert weights["FINN"] == pytest.approx((3500 * 31.90) / expected_total, rel=1e-6)
    assert weights["FINN"] < 0.25  # nowhere near the reported ~99.5%


def test_finn_position_uses_correct_quote_currency(fake_yfinance):
    quotes = fetch_quotes([h.ticker for h in PORTFOLIO])
    snapshot = build_snapshot(PORTFOLIO, quotes, account_currency="CAD", usd_cad=1.36)
    finn_position = next(p for p in snapshot.positions if p.ticker == "FINN")
    assert finn_position.currency == "CAD"
    assert finn_position.market_value == pytest.approx(3500 * 31.90)


def test_us_securities_still_resolve_correctly_alongside_canadian_ones(fake_yfinance):
    quotes = fetch_quotes([h.ticker for h in PORTFOLIO])
    snapshot = build_snapshot(PORTFOLIO, quotes, account_currency="CAD", usd_cad=1.36)
    nvda = next(p for p in snapshot.positions if p.ticker == "NVDA")
    vht = next(p for p in snapshot.positions if p.ticker == "VHT")
    assert nvda.currency == "USD" and nvda.market_value == pytest.approx(150 * 224.41 * 1.36)
    assert vht.currency == "USD" and vht.market_value == pytest.approx(100 * 324.47 * 1.36)


def test_mismatched_identity_fails_closed_end_to_end(monkeypatch):
    """If the mapped Canadian symbol ever returned data for the wrong
    currency (e.g. a broken mapping or a delisting), fetch_quotes must
    surface an unresolved identity quote -- never a fabricated value."""
    import yfinance

    class _CorruptedTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="10d", interval="1d", auto_adjust=False):
            return pd.DataFrame({"Close": [31.60, 31.90]}, index=pd.date_range("2026-01-01", periods=2))

        @property
        def fast_info(self):
            return {"currency": "USD", "quoteType": "EQUITY"}  # simulated data corruption: should be CAD/ETF

    monkeypatch.setattr(yfinance, "Ticker", _CorruptedTicker)
    quotes = fetch_quotes(["FINN"])
    assert quotes["FINN"].status.startswith("identity:")
    assert quotes["FINN"].price is None
    assert "FINN" in quotes["FINN"].status


def test_unknown_ticker_fails_closed(fake_yfinance):
    quotes = fetch_quotes(["TOTALLYMADEUPTICKERXYZ"])
    quote = quotes["TOTALLYMADEUPTICKERXYZ"]
    assert quote.price is None
    assert quote.status.startswith("error")


def test_build_snapshot_surfaces_identity_error_as_position_issue():
    """core.analytics.build_snapshot must carry the exact fail-closed
    message through to Position.issues so the app layer can block
    analysis with the compact message, not a generic notice."""
    from core.models import Quote

    quotes = {"FINN": Quote("FINN", None, None, None, status="identity:FINN 无法可靠识别证券身份，请确认代码或交易市场。")}
    snapshot = build_snapshot([HoldingInput("FINN", 100)], quotes, cash=1000, account_currency="CAD")
    position = snapshot.positions[0]
    assert position.status == "incomplete"
    assert position.market_value is None
    assert any("无法可靠识别证券身份" in issue for issue in position.issues)
