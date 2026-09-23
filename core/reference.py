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
    # Step 2B.1 coverage expansion -- official issuer factsheet, "Top 10
    # holdings (% of net asset value)", source: Vanguard Investments Canada
    # Inc., https://fund-docs.vanguard.com/VDY_FTSE_Canadian_High_Dividend_
    # Yield_Index_ETF_9560_FS_EN_CA.pdf ("Factsheet | July 31, 2026",
    # doc ref F9330EN_CA_072026). Top 10 total per the factsheet: 69.3%.
    # Canadian-listed constituents, bare tickers per this project's existing
    # ETF_HOLDINGS convention (no exchange suffix, matching how US
    # constituents are stored above).
    "VDY": {
        "RY": .159, "TD": .109, "BMO": .069, "ENB": .064, "CM": .059, "BNS": .059,
        "CNQ": .053, "SU": .043, "MFC": .040, "TRP": .038,
    },
    # Step 2B.1 coverage expansion -- official issuer factsheet, "Ten
    # largest holdings and % of total net assets", source: The Vanguard
    # Group, Inc., https://workplace.vanguard.com/assets/corp/
    # fund_communications/pdf_publish/us-products/fact-sheet/F0956.pdf
    # ("As of June 30, 2026", doc ref F0956_062026). Top 10 total per the
    # factsheet: 52.3%.
    "VHT": {
        "LLY": .142, "JNJ": .089, "ABBV": .066, "UNH": .056, "MRK": .047,
        "AMGN": .029, "TMO": .027, "ABT": .023, "GILD": .023, "ISRG": .021,
    },
}

# Step 2B.1 audit finding: FINN (Fidelity Global Innovators ETF Series L)
# remains deliberately UNCOVERED. Fidelity's own official factsheet
# (https://www.fidelity.ca/content/dam/fidelity/en/documents/etf/
# fact-sheet-finn-en.pdf, "Top 10 holdings as at June 30, 2026") publishes
# only the ranked constituent names plus one aggregate "Top ten holdings
# aggregate 57.4%" figure -- no individual per-holding weight is disclosed
# anywhere in the official document. Fabricating a per-holding split (even
# an even split of 57.4%) would be an invented weight, not verified data --
# fail closed instead per this step's sourcing rule. Revisit only if
# Fidelity ever publishes individual constituent weights.

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
# allocation logic keys off of; only the display string is localized here.
#
# Step 2B.1 Asset Classification V2: "Fixed Income" and "Cash-like" are now
# separate allocation buckets (core.analytics.analyze splits them using this
# same CASH_LIKE_TICKERS set -- SGOV/CBIL are ultra-short-duration Treasury
# instruments economically closer to cash than to duration-bearing bonds;
# XSB/SHY/AGG/BND/TLT remain "Fixed Income"). Before this step both were
# combined under one "Bonds & Cash-like" label, hiding the structural
# difference -- see the Step 2B.1 report. This is classification only: no
# duration/correlation/hedge claim is implied by either label.
ASSET_CLASS_LABELS_ZH = {
    "Equity": "股票", "Fixed Income": "固定收益", "Cash-like": "现金类", "Cash": "现金", "Other": "其他",
}
ASSET_CLASS_LABELS_EN = {
    "Equity": "Equity", "Fixed Income": "Fixed Income", "Cash-like": "Cash-like", "Cash": "Cash", "Other": "Other",
}
