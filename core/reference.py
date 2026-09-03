"""Frozen reference data adapted from the verified legacy system at commit
6308824bd165174a89f1928ceb0bf7d9bad49c14.

ETF data is a 2025-06-30 Top-N snapshot, not live or exhaustive. An ETF
absent from ETF_HOLDINGS is unresolved; the analytics engine never guesses.
"""

ETF_HOLDINGS_AS_OF = "2025-06-30"

ETF_HOLDINGS: dict[str, dict[str, float]] = {
    "VOO": {"AAPL": .070, "MSFT": .066, "NVDA": .066, "AMZN": .038, "META": .026, "GOOGL": .021, "GOOG": .018, "AVGO": .017, "TSLA": .016, "JPM": .013, "LLY": .011, "V": .010, "UNH": .008, "XOM": .008, "BRK-B": .017},
    "SPY": {"AAPL": .070, "MSFT": .066, "NVDA": .066, "AMZN": .038, "META": .026, "GOOGL": .021, "GOOG": .018, "AVGO": .017, "TSLA": .016, "JPM": .013, "LLY": .011, "V": .010, "UNH": .008, "XOM": .008, "BRK-B": .017},
    "IVV": {"AAPL": .070, "MSFT": .066, "NVDA": .066, "AMZN": .038, "META": .026, "GOOGL": .021, "GOOG": .018, "AVGO": .017, "TSLA": .016, "JPM": .013, "LLY": .011, "V": .010, "UNH": .008, "XOM": .008, "BRK-B": .017},
    # ZSP (BMO S&P 500 Index ETF) replicates the same S&P 500 index as
    # VOO/SPY/IVV above, just CAD-priced -- same constituents, same weights,
    # not a separately-sourced or guessed snapshot.
    "ZSP": {"AAPL": .070, "MSFT": .066, "NVDA": .066, "AMZN": .038, "META": .026, "GOOGL": .021, "GOOG": .018, "AVGO": .017, "TSLA": .016, "JPM": .013, "LLY": .011, "V": .010, "UNH": .008, "XOM": .008, "BRK-B": .017},
    "VTI": {"AAPL": .061, "MSFT": .058, "NVDA": .058, "AMZN": .033, "META": .023, "GOOGL": .018, "GOOG": .016, "AVGO": .015, "TSLA": .015, "JPM": .012, "BRK-B": .015},
    "QQQ": {"AAPL": .089, "MSFT": .086, "NVDA": .086, "AMZN": .056, "AVGO": .045, "META": .038, "GOOGL": .028, "GOOG": .026, "TSLA": .025, "NFLX": .024, "AMD": .010},
    "QQQM": {"AAPL": .089, "MSFT": .086, "NVDA": .086, "AMZN": .056, "AVGO": .045, "META": .038, "GOOGL": .028, "GOOG": .026, "TSLA": .025, "NFLX": .024, "AMD": .010},
    "XLK": {"AAPL": .150, "MSFT": .130, "NVDA": .120, "AVGO": .065, "AMD": .020},
    "SMH": {"NVDA": .200, "AVGO": .090, "AMD": .045},
    "SOXX": {"NVDA": .090, "AVGO": .070, "AMD": .045},
    "VUG": {"AAPL": .095, "MSFT": .090, "NVDA": .090, "AMZN": .060, "META": .040, "AVGO": .035, "GOOGL": .030, "GOOG": .026, "TSLA": .025, "NFLX": .020},
    "MTUM": {"NVDA": .060, "META": .045, "AVGO": .040, "NFLX": .035, "JPM": .030},
    "SPMO": {"NVDA": .050, "AVGO": .045, "JPM": .040, "META": .035, "TSLA": .030},
}

ETF_META = {
    "VOO": ("Broad Market", "Equity"), "SPY": ("Broad Market", "Equity"),
    "IVV": ("Broad Market", "Equity"), "VTI": ("Broad Market", "Equity"),
    "QQQ": ("Technology", "Equity"), "QQQM": ("Technology", "Equity"),
    "XLK": ("Technology", "Equity"), "SMH": ("Technology", "Equity"),
    "SOXX": ("Technology", "Equity"), "SPMO": ("Multi-Sector", "Equity"),
    "MTUM": ("Multi-Sector", "Equity"), "VUG": ("Multi-Sector", "Equity"),
    "VTV": ("Multi-Sector", "Equity"), "SCHD": ("Multi-Sector", "Equity"),
    "VYM": ("Multi-Sector", "Equity"), "VHT": ("Healthcare", "Equity"),
    "XLV": ("Healthcare", "Equity"), "XLF": ("Financials", "Equity"),
    "XLE": ("Energy", "Equity"), "XLI": ("Industrials", "Equity"),
    "XLU": ("Utilities", "Equity"), "XLP": ("Consumer Staples", "Equity"),
    "XLY": ("Consumer Discretionary", "Equity"), "GRID": ("Infrastructure", "Equity"),
    "VXUS": ("International", "Equity"), "VEA": ("International", "Equity"),
    "VWO": ("International", "Equity"), "EFA": ("International", "Equity"),
    "ZSP": ("Broad Market", "Equity"), "FINN": ("International", "Equity"),
    "VDY": ("Multi-Sector", "Equity"),
    "SGOV": ("Fixed Income", "Fixed Income"), "BIL": ("Fixed Income", "Fixed Income"),
    "SHY": ("Fixed Income", "Fixed Income"), "AGG": ("Fixed Income", "Fixed Income"),
    "BND": ("Fixed Income", "Fixed Income"), "TLT": ("Fixed Income", "Fixed Income"),
    "CBIL": ("Fixed Income", "Fixed Income"), "XSB": ("Fixed Income", "Fixed Income"),
    "GLD": ("Commodities", "Other"), "USO": ("Commodities", "Other"),
}

STOCK_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "GOOG": "Technology", "META": "Technology",
    "AVGO": "Technology", "AMD": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
    "JPM": "Financials", "V": "Financials", "BRK-B": "Financials",
    "UNH": "Healthcare", "LLY": "Healthcare", "XOM": "Energy",
}

CASH_LIKE_TICKERS = {"SGOV", "CBIL"}

# Chinese-first display names for asset-class keys used throughout core/analytics.py
# and core/models.py. Keys are the canonical English identifiers the scoring and
# allocation logic keys off of; only the display string is localized here. The
# "Fixed Income" bucket keeps its canonical key (scoring/thresholds are
# unchanged) -- only its user-facing label reads "Bonds & Cash-like", since
# it holds bonds, T-bills, and other cash-like instruments (e.g. CBIL, XSB,
# SGOV), not literally "cash" (which is its own separate bucket).
ASSET_CLASS_LABELS_ZH = {
    "Equity": "股票", "Fixed Income": "债券与现金类", "Cash": "现金", "Other": "其他",
}
ASSET_CLASS_LABELS_EN = {
    "Equity": "Equity", "Fixed Income": "Bonds & Cash-like", "Cash": "Cash", "Other": "Other",
}
