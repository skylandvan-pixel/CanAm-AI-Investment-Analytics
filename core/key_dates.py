"""Step 2A.5 -- Security Key Dates.

Layer 1, deterministic "what important, already-known future date should the
user watch for this security" reference for the Page 1 Security Profile
card. No LLM call anywhere in this module, ever: Gemini/Anthropic must never
invent, estimate, or infer an event date (see the Step 2A.3/2A.4 report-
integrity guardrails, which this module never touches or feeds into).

V1 supports exactly two event types, in descending order of reliability:
  1. next company earnings date -- individual stocks only, via the existing
     yfinance market-data dependency (core.market already depends on it for
     quotes; no new dependency added here).
  2. next FOMC rate decision -- a small maintained local reference table of
     published Federal Reserve meeting dates (see _FOMC_DECISION_DATES),
     never a network call.

ETF rebalance/reconstitution dates and a "major earnings window" concept
were both audited and deliberately deferred: no reliable, already-available
schedule metadata exists in this project for either, and building one from
scratch is out of scope for this patch (see the Step 2A.5 report).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Literal

from core.security_profile import resolve_security_profile
from core.ticker_resolution import resolve_ticker

# Hard scope limits (see the Step 2A.5 spec) -- never events further out than
# this, never more than this many shown at once.
_HORIZON_DAYS = 90
_MAX_EVENTS = 3

EventStatus = Literal["confirmed", "estimated", "window"]


@dataclass(frozen=True)
class SecurityKeyDate:
    event_date: date
    event_type: str  # "earnings" | "fomc"
    title_zh: str
    status: EventStatus
    relevance_note: str
    source_kind: str  # "yfinance" | "fomc_reference"
    window_end: date | None = None  # only set when status == "window"


# Published Federal Reserve FOMC decision/statement dates -- the SECOND day
# of each two-day meeting, when the Committee actually releases its policy
# statement (the Fed publishes a statement at 2pm ET on the second day; the
# first day has no decision). Cross-checked against
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm and the
# Fed's own per-meeting pages (e.g. .../fomcpresconf20260617.htm) in
# September 2026. 2027 dates are the Fed's own "tentative" schedule,
# announced 2025-09-05 -- extend this tuple before it runs out, from the
# same source. A local reference table, never a network call and never
# Gemini-generated.
_FOMC_DECISION_DATES: tuple[date, ...] = (
    date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 9),
    date(2027, 7, 28), date(2027, 9, 15), date(2027, 10, 27), date(2027, 12, 8),
)


def next_fomc_decision_date(today: date) -> date | None:
    """The nearest published FOMC decision date with today <= date <= today
    + 90 days, or None if the reference table has no such date (fail
    closed -- never guessed or extrapolated)."""
    horizon = today + timedelta(days=_HORIZON_DAYS)
    upcoming = [d for d in _FOMC_DECISION_DATES if today <= d <= horizon]
    return min(upcoming) if upcoming else None


def etf_is_us_market_exposed(category_value: str, subcategory_value: str) -> bool:
    """Deterministic FOMC-relevance rule for ETFs, built entirely from
    fields core.security_profile already exposes -- no new classification
    data. Matches against the specific category_value values in use (never
    a substring scan of free-text description fields: several international
    funds' own subcategory labels literally contain the characters "美国" as
    part of a NEGATED phrase, e.g. "全球（除美国）股票指数 ETF" / "发达市场
    （除美加）股票指数 ETF" -- ex-US/ex-US-and-Canada -- so a bare `"美国" in
    text` substring check would wrongly mark those as U.S.-exposed).

    "美国股票"/"美国股票（加元计价）" (the latter is ZSP: CAD-priced but
    S&P 500-tracking, genuinely U.S.-market exposed) are always U.S.-exposed.
    "固定收益" mixes genuinely U.S. Treasury ETFs (SGOV, BIL, ...) with
    Canadian bond ETFs (CBIL, XSB) in one bucket -- disambiguate via the
    fund-category label's own prefix ("美国..." vs "加拿大..."), never a
    substring match. Every other category (Canadian equity, international,
    global, alternative) is deliberately omitted rather than overreached."""
    if category_value in {"美国股票", "美国股票（加元计价）"}:
        return True
    if category_value == "固定收益":
        return subcategory_value.startswith("美国")
    return False


def next_earnings_date(
    ticker: str, *, today: date,
    fetch_calendar: Callable[[str], dict] | None = None,
) -> SecurityKeyDate | None:
    """Individual-stock next earnings date only -- callers must never invoke
    this for an ETF (see get_security_key_dates). Fails closed (returns
    None) on any provider error, missing/malformed data, a past date, or a
    date outside the 90-day horizon -- never a guessed/substituted date, and
    never a raw provider error surfaced to the caller.

    fetch_calendar defaults to yfinance's own Ticker(...).calendar (the
    existing project dependency -- core.market already depends on yfinance
    for quotes; no new dependency added) and is injectable purely for
    deterministic testing without a live network call."""
    if fetch_calendar is None:
        def fetch_calendar(resolved_ticker: str) -> dict:
            import yfinance as yf
            return yf.Ticker(resolved_ticker).calendar or {}

    try:
        # Canonical ticker identity (core.ticker_resolution), the same
        # single source of truth core.market.fetch_quotes uses -- never a
        # second, independently-drifting notion of "which market symbol
        # does this ticker mean."
        identity = resolve_ticker(ticker)
        calendar = fetch_calendar(identity.resolved_ticker)
        raw_dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
        if not raw_dates:
            return None
        # yfinance's own calendar shape: a list of 1 date (a single known
        # date) or 2 dates (an estimated date-range window). Any other
        # shape is unrecognized and fails closed rather than guessed at.
        normalized = sorted({d for d in raw_dates if isinstance(d, date)})
        if not normalized or len(normalized) > 2:
            return None
        if len(normalized) == 1:
            event_date, window_end, status = normalized[0], None, "estimated"
        else:
            event_date, window_end, status = normalized[0], normalized[1], "window"
        horizon = today + timedelta(days=_HORIZON_DAYS)
        if not (today <= event_date <= horizon):
            return None
        return SecurityKeyDate(
            event_date, "earnings", f"{ticker} 下一季度财报", status,
            "关注是否改变当前集中度与减仓/持有判断", "yfinance", window_end,
        )
    except Exception:
        # Fail closed: provider error, network failure, unexpected shape --
        # never a guessed date, never a raw exception surfaced to the UI.
        return None


def _finalize(events: list[SecurityKeyDate], today: date) -> list[SecurityKeyDate]:
    """The shared sorting/deduplication pipeline (Step 2A.5 spec): drop
    invalid/past events, drop anything beyond the 90-day horizon,
    deduplicate same (event_type, date), sort ascending by date, keep at
    most _MAX_EVENTS. Applied regardless of which event type produced a
    given entry, so future event types inherit the same guarantees."""
    horizon = today + timedelta(days=_HORIZON_DAYS)
    seen: set[tuple[str, date]] = set()
    kept: list[SecurityKeyDate] = []
    for event in events:
        if not (today <= event.event_date <= horizon):
            continue
        key = (event.event_type, event.event_date)
        if key in seen:
            continue
        seen.add(key)
        kept.append(event)
    kept.sort(key=lambda e: e.event_date)
    return kept[:_MAX_EVENTS]


def get_security_key_dates(
    ticker: str, *, today: date | None = None,
    fetch_calendar: Callable[[str], dict] | None = None,
) -> list[SecurityKeyDate]:
    """The single entry point the UI calls (Page 1 Security Profile card).
    Deterministic and network-light: FOMC is a pure local table lookup;
    earnings (individual stocks only) makes at most one yfinance call. Never
    calls Gemini/Anthropic in any way, and never requires Pro/Beta unlock.

    Relevance rules (Step 2A.5 spec): an individual stock gets its own
    earnings date plus FOMC (the current Security Profile snapshot is
    entirely U.S.-listed large caps); an ETF gets FOMC only when
    etf_is_us_market_exposed says so, and never a constituent's earnings
    date merely because it might be held inside the ETF."""
    today = today or date.today()
    profile = resolve_security_profile(ticker)
    if profile is None:
        return []
    events: list[SecurityKeyDate] = []
    is_stock = profile.kind_label == "股票（Stock）"
    if is_stock:
        earnings = next_earnings_date(ticker, today=today, fetch_calendar=fetch_calendar)
        if earnings is not None:
            events.append(earnings)
        fomc_relevant = True
    else:
        fomc_relevant = etf_is_us_market_exposed(profile.category_value, profile.subcategory_value)
    if fomc_relevant:
        fomc_date = next_fomc_decision_date(today)
        if fomc_date is not None:
            events.append(SecurityKeyDate(
                fomc_date, "fomc", "美联储 FOMC 利率决议", "confirmed",
                "关注利率路径对相关资产估值的影响", "fomc_reference",
            ))
    return _finalize(events, today)


def _month_third(day: int) -> str:
    if day <= 10:
        return "上旬"
    if day <= 20:
        return "中旬"
    return "下旬"


def format_event_date(event: SecurityKeyDate) -> str:
    """User-facing Chinese date text -- never false precision. CONFIRMED
    shows a plain ISO date; ESTIMATED is prefixed "预计"; WINDOW collapses
    the date range to a month + third-of-month bucket (e.g. "预计 10月下旬"),
    never a fabricated single day."""
    if event.status == "confirmed":
        return event.event_date.isoformat()
    if event.status == "estimated":
        return f"预计 {event.event_date.isoformat()}"
    if event.status == "window":
        anchor_day = event.event_date.day
        if event.window_end is not None:
            anchor_day = (event.event_date.day + event.window_end.day) // 2
        return f"预计 {event.event_date.month}月{_month_third(anchor_day)}"
    return event.event_date.isoformat()
