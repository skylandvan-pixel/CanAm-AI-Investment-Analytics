"""Step 2A.5/2A.7 -- Security Key Dates.

Layer 1, deterministic "what important, already-known future date should the
user watch for this security" reference for the Page 1 Security Profile
card. No LLM call anywhere in this module, ever: Gemini/Anthropic must never
invent, estimate, or infer an event date (see the Step 2A.3/2A.4 report-
integrity guardrails, which this module never touches or feeds into).

V2 (Step 2A.7) supports exactly four event types:
  1. next company earnings date -- individual stocks only, via the existing
     yfinance market-data dependency (core.market already depends on it for
     quotes; no new dependency added here).
  2. next FOMC rate decision -- a small maintained local reference table of
     published Federal Reserve meeting dates (see _FOMC_DECISION_DATES).
  3. next U.S. CPI release -- a small maintained local reference table of
     officially published BLS Consumer Price Index release dates (see
     _CPI_RELEASE_DATES).
  4. next U.S. Nonfarm Payrolls / Employment Situation release -- a small
     maintained local reference table of officially published BLS release
     dates (see _NFP_RELEASE_DATES).
All three macro tables are local, never a network call and never Gemini-
generated -- see each table's own comment for its official source.

ETF rebalance/reconstitution dates and a "major earnings window" concept
were both audited and deliberately deferred: no reliable, already-available
schedule metadata exists in this project for either, and building one from
scratch is out of scope for this project (see the Step 2A.5 report). Step
2A.7 deliberately adds ONLY CPI and NFP to the macro-event pool -- no PCE,
GDP, retail sales, ISM, jobless claims, Fed speeches, Treasury auctions, or
generic economic-calendar events (see the Step 2A.7 spec).
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
    event_type: str  # "earnings" | "fomc" | "cpi" | "nfp"
    title_zh: str
    status: EventStatus
    relevance_note: str
    source_kind: str  # "yfinance" | "fomc_reference" | "cpi_reference" | "nfp_reference"
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

# Officially published U.S. Bureau of Labor Statistics Consumer Price Index
# release dates -- source: https://www.bls.gov/schedule/news_release/cpi.htm
# (re-verified directly against the live official schedule table in
# September 2026; every date below matches that table row for row). The
# October 2025 report was not published -- per BLS's own official notice at
# https://www.bls.gov/bls/news-release/cpi.htm: "October 2025 Consumer Price
# Index -- Not published because of 2025 lapse in federal government
# appropriations" -- so no CPI release date exists for that reference month
# and none is included here. A local reference table, never a network call
# and never Gemini-generated -- extend it before it runs out, from the same
# official BLS schedule page.
_CPI_RELEASE_DATES: tuple[date, ...] = (
    date(2025, 12, 18),
    date(2026, 1, 13), date(2026, 2, 13), date(2026, 3, 11), date(2026, 4, 10),
    date(2026, 5, 12), date(2026, 6, 10), date(2026, 7, 14), date(2026, 8, 12),
    date(2026, 9, 11), date(2026, 10, 14), date(2026, 11, 10), date(2026, 12, 10),
)

# Officially published U.S. Bureau of Labor Statistics Employment Situation
# (nonfarm payrolls / NFP) release dates -- source:
# https://www.bls.gov/schedule/news_release/empsit.htm (re-verified directly
# against the live official schedule table in September 2026; every date
# below matches that table row for row). The 2025-12-16 entry (November 2025
# reference month) was specifically re-audited: it is exactly what BLS's own
# schedule table publishes for that reference month, carries no rescheduling
# footnote on the BLS page itself, and is consistent with BLS's official
# notice (https://www.bls.gov/bls/news-release/empsit.htm) that the prior
# reference month's release ("October 2025 Employment Situation") was "Not
# published because of 2025 lapse in federal government appropriations" --
# i.e. 2025-12-16 is the officially published date for November 2025 data,
# not a heuristic or third-party-inferred value. A local reference table,
# never a network call and never Gemini-generated -- extend it before it
# runs out, from the same official BLS schedule page.
_NFP_RELEASE_DATES: tuple[date, ...] = (
    date(2025, 12, 16),
    date(2026, 1, 9), date(2026, 2, 11), date(2026, 3, 6), date(2026, 4, 3),
    date(2026, 5, 8), date(2026, 6, 5), date(2026, 7, 2), date(2026, 8, 7),
    date(2026, 9, 4), date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
)


def _next_official_date(reference_dates: tuple[date, ...], today: date) -> date | None:
    """Shared lookup for every local official-date reference table (FOMC,
    CPI, NFP): the nearest date with today <= date <= today + 90 days, or
    None if the table has no such date -- fail closed, never guessed or
    extrapolated beyond the officially published schedule."""
    horizon = today + timedelta(days=_HORIZON_DAYS)
    upcoming = [d for d in reference_dates if today <= d <= horizon]
    return min(upcoming) if upcoming else None


def next_fomc_decision_date(today: date) -> date | None:
    """The nearest published FOMC decision date within the 90-day horizon,
    or None (fail closed -- never guessed or extrapolated)."""
    return _next_official_date(_FOMC_DECISION_DATES, today)


def next_cpi_release_date(today: date) -> date | None:
    """The nearest officially published BLS CPI release date within the
    90-day horizon, or None (fail closed -- never guessed or extrapolated;
    never a "second Tuesday of the month"-style heuristic)."""
    return _next_official_date(_CPI_RELEASE_DATES, today)


def next_nfp_release_date(today: date) -> date | None:
    """The nearest officially published BLS Employment Situation (NFP)
    release date within the 90-day horizon, or None (fail closed -- never
    guessed or extrapolated; never a "first Friday of the month"-style
    heuristic)."""
    return _next_official_date(_NFP_RELEASE_DATES, today)


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
            "关注业绩与指引是否改变当前持有判断", "yfinance", window_end,
        )
    except Exception:
        # Fail closed: provider error, network failure, unexpected shape --
        # never a guessed date, never a raw exception surfaced to the UI.
        return None


# Step 2A.7: when more than _MAX_EVENTS candidates are eligible, priority
# decides which ones SURVIVE the cut -- it has no effect on display order
# (events are always rendered chronologically once selected, see
# _finalize). A security's own earnings date is the single most decision-
# relevant fact this module can offer, so it always outranks the three
# macro releases; among macro releases, FOMC (a policy decision) outranks
# CPI, which outranks NFP, per the Step 2A.7 spec.
_EVENT_TYPE_PRIORITY = {"earnings": 0, "fomc": 1, "cpi": 2, "nfp": 3}


def _finalize(events: list[SecurityKeyDate], today: date) -> list[SecurityKeyDate]:
    """The shared sorting/deduplication/selection pipeline: drop invalid/
    past events, drop anything beyond the 90-day horizon, deduplicate same
    (event_type, date), then SELECT the top _EVENT_TYPE_PRIORITY-ranked
    events up to _MAX_EVENTS, and finally sort just that selection
    chronologically for display. Priority governs only which events survive
    the cut, never the rendered order -- see _EVENT_TYPE_PRIORITY."""
    horizon = today + timedelta(days=_HORIZON_DAYS)
    seen: set[tuple[str, date]] = set()
    valid: list[SecurityKeyDate] = []
    for event in events:
        if not (today <= event.event_date <= horizon):
            continue
        key = (event.event_type, event.event_date)
        if key in seen:
            continue
        seen.add(key)
        valid.append(event)
    valid.sort(key=lambda e: _EVENT_TYPE_PRIORITY.get(e.event_type, 99))
    selected = valid[:_MAX_EVENTS]
    selected.sort(key=lambda e: e.event_date)
    return selected


def get_security_key_dates(
    ticker: str, *, today: date | None = None,
    fetch_calendar: Callable[[str], dict] | None = None,
) -> list[SecurityKeyDate]:
    """The single entry point the UI calls (Page 1 Security Profile card).
    Deterministic and network-light: FOMC/CPI/NFP are pure local table
    lookups; earnings (individual stocks only) makes at most one yfinance
    call. Never calls Gemini/Anthropic in any way, and never requires
    Pro/Beta unlock.

    Relevance rules (Step 2A.5, extended by Step 2A.7): an individual stock
    gets its own earnings date plus all three macro releases (the current
    Security Profile snapshot is entirely U.S.-listed large caps); an ETF
    gets the three macro releases only when etf_is_us_market_exposed says
    so, and never a constituent's earnings date merely because it might be
    held inside the ETF. FOMC/CPI/NFP always share the same relevance gate
    -- Step 2A.7 deliberately reuses the existing U.S.-market-exposure rule
    rather than inventing a separate one per macro release."""
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
        macro_relevant = True
    else:
        macro_relevant = etf_is_us_market_exposed(profile.category_value, profile.subcategory_value)
    if macro_relevant:
        fomc_date = next_fomc_decision_date(today)
        if fomc_date is not None:
            events.append(SecurityKeyDate(
                fomc_date, "fomc", "美联储 FOMC 利率决议", "confirmed",
                "关注利率路径对相关资产估值的影响", "fomc_reference",
            ))
        cpi_date = next_cpi_release_date(today)
        if cpi_date is not None:
            events.append(SecurityKeyDate(
                cpi_date, "cpi", "美国 CPI 通胀数据", "confirmed",
                "关注通胀变化对利率预期与资产估值的影响", "cpi_reference",
            ))
        nfp_date = next_nfp_release_date(today)
        if nfp_date is not None:
            events.append(SecurityKeyDate(
                nfp_date, "nfp", "美国非农就业报告", "confirmed",
                "关注就业变化对货币政策预期的影响", "nfp_reference",
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
