from __future__ import annotations

from core.models import HoldingInput, Quote

DEMO_HOLDINGS = [
    HoldingInput("VOO", 18, 480, "USD"),
    HoldingInput("XLK", 25, 205, "USD"),
    HoldingInput("NVDA", 55, 112, "USD"),
    HoldingInput("SGOV", 70, 100, "USD"),
    HoldingInput("AAPL", 18, 215, "USD"),
]

# A clearly-labelled, internally consistent demo snapshot. It never enters
# the live-data path and is not represented as current market data.
DEMO_QUOTES = {
    "VOO": Quote("VOO", 528.40, "USD", 526.30, "Demo snapshot", "demo"),
    "XLK": Quote("XLK", 237.10, "USD", 238.20, "Demo snapshot", "demo"),
    "NVDA": Quote("NVDA", 132.60, "USD", 130.90, "Demo snapshot", "demo"),
    "SGOV": Quote("SGOV", 100.52, "USD", 100.50, "Demo snapshot", "demo"),
    "AAPL": Quote("AAPL", 232.20, "USD", 230.80, "Demo snapshot", "demo"),
}
