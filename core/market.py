from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime, timezone

from core.models import Quote
from core.ticker_resolution import TickerIdentityError, resolve_ticker, validate_identity


def _check_price_integrity(price: float, currency: str | None) -> None:
    """Deterministic data-integrity guard, not a financial heuristic: a
    verified quote must have a finite, positive price and a known
    currency before it is allowed into analytics. This catches corrupt
    or nonsensical market data (e.g. a zero/NaN close, a missing
    currency) -- it does not judge whether the price or the resulting
    position is "reasonable"."""
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Quoted price is not a finite positive number")
    if not currency:
        raise ValueError("Security currency unavailable")


def fetch_usd_cad_rate() -> float | None:
    """Return a verified USD→CAD close, or None. Never substitutes a constant."""
    import yfinance as yf
    try:
        history = yf.Ticker("CAD=X").history(period="10d", interval="1d", auto_adjust=False)
        closes = history["Close"].dropna() if not history.empty and "Close" in history else []
        value = float(closes.iloc[-1]) if len(closes) else None
        return value if value and value > 0 else None
    except Exception:
        return None


def fetch_quotes(tickers: Iterable[str]) -> dict[str, Quote]:
    """Fetch verified previous closes. Provider failure, an unresolved
    ticker, or a resolved-but-mismatched security identity all return an
    unresolved Quote for that ticker -- never a substituted or guessed
    price. Results are always keyed by the user's input ticker; only the
    underlying query targets the resolved market symbol (e.g. bare
    "FINN" queries "FINN.NE" internally but is still returned as "FINN")."""
    import yfinance as yf

    result: dict[str, Quote] = {}
    for raw in sorted(set(tickers)):
        ticker = raw.upper()
        try:
            identity = resolve_ticker(ticker)
            instrument = yf.Ticker(identity.resolved_ticker)
            history = instrument.history(period="10d", interval="1d", auto_adjust=False)
            closes = history["Close"].dropna() if not history.empty and "Close" in history else []
            if len(closes) < 1:
                raise ValueError("No completed close available")
            price = float(closes.iloc[-1])
            previous = float(closes.iloc[-2]) if len(closes) > 1 else None
            info = getattr(instrument, "fast_info", {}) or {}
            currency = info.get("currency")
            quote_type = info.get("quoteType")
            validate_identity(identity, currency=currency, quote_type=quote_type)
            if currency is None and identity.expected_currency:
                currency = identity.expected_currency
            _check_price_integrity(price, currency)
            result[ticker] = Quote(
                ticker, price, currency, previous,
                str(getattr(closes.index[-1], "date", lambda: closes.index[-1])()), "verified",
            )
        except TickerIdentityError as exc:
            result[ticker] = Quote(
                ticker, None, None, None,
                datetime.now(timezone.utc).isoformat(timespec="seconds"), f"identity:{exc}",
            )
        except Exception as exc:  # fail closed: no substituted or guessed price
            result[ticker] = Quote(
                ticker, None, None, None,
                datetime.now(timezone.utc).isoformat(timespec="seconds"), f"error:{type(exc).__name__}",
            )
    return result
