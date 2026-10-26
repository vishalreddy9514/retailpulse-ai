"""The retailer's trading calendar.

The data shows the shop does not trade on Saturdays, England & Wales bank holidays,
Easter Sunday or over Christmas. Only one Saturday (2009-12-05) in two years has sales.
Forecasts must be made for *future* dates, so they cannot rely on "a day that had sales".
They need a rule that is known in advance. ``is_expected_trading_day`` is that rule.
``compare_with_observed`` measures how well it matches the history. The result is
recorded in the aggregation report.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import holidays
import pandas as pd
from dateutil.easter import easter

SATURDAY = 5
# Observed Christmas shutdown: no trading from 24 December to 3 January inclusive.
SHUTDOWN_START = (12, 24)
SHUTDOWN_END = (1, 3)


@lru_cache(maxsize=32)
def _bank_holidays(year: int) -> frozenset[date]:
    return frozenset(holidays.country_holidays("GB", subdiv="ENG", years=year).keys())


def closure_reason(day: date | pd.Timestamp) -> str | None:
    """Why the shop is expected to be closed on ``day``, or None if it should trade."""
    d = pd.Timestamp(day).date()
    if d.weekday() == SATURDAY:
        return "saturday"
    if (d.month, d.day) >= SHUTDOWN_START or (d.month, d.day) <= SHUTDOWN_END:
        return "christmas_shutdown"
    if d in _bank_holidays(d.year):
        return "bank_holiday"
    if d == easter(d.year):
        return "easter_sunday"
    return None


def is_expected_trading_day(day: date | pd.Timestamp) -> bool:
    return closure_reason(day) is None


def expected_trading_days(start: date | pd.Timestamp, periods: int) -> list[pd.Timestamp]:
    """The next ``periods`` expected trading days strictly after ``start``."""
    out: list[pd.Timestamp] = []
    day = pd.Timestamp(start).normalize()
    while len(out) < periods:
        day += pd.Timedelta(days=1)
        if is_expected_trading_day(day):
            out.append(day)
    return out


def build_calendar(start: date | pd.Timestamp, end: date | pd.Timestamp) -> pd.DataFrame:
    """One row per calendar day with the rule-based trading flag and calendar fields."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    cal = pd.DataFrame({"date": days})
    cal["closure_reason"] = pd.array([closure_reason(d) for d in days], dtype="string")
    cal["is_expected_trading_day"] = cal["closure_reason"].isna()
    cal["day_of_week"] = cal["date"].dt.day_of_week.astype("int8")
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype("int8")
    cal["month"] = cal["date"].dt.month.astype("int8")
    cal["day_of_month"] = cal["date"].dt.day.astype("int8")
    return cal


def compare_with_observed(observed_trading_days: pd.Series) -> dict:
    """How the rule agrees with the days that actually had sales."""
    observed = set(pd.to_datetime(observed_trading_days).dt.normalize())
    cal = build_calendar(min(observed), max(observed))
    expected = set(cal.loc[cal["is_expected_trading_day"], "date"])
    traded_but_closed = sorted(observed - expected)
    open_but_no_sales = sorted(expected - observed)
    return {
        "days_checked": len(cal),
        "agreement_rate": round(
            1 - (len(traded_but_closed) + len(open_but_no_sales)) / len(cal), 4
        ),
        "traded_on_expected_closed_day": [str(d.date()) for d in traded_but_closed],
        "no_sales_on_expected_trading_day": [str(d.date()) for d in open_but_no_sales],
    }
