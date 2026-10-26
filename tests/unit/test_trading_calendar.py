from __future__ import annotations

import pandas as pd
import pytest

from src.processing.trading_calendar import (
    build_calendar,
    closure_reason,
    compare_with_observed,
    expected_trading_days,
)


@pytest.mark.parametrize(
    ("day", "reason"),
    [
        ("2011-03-05", "saturday"),
        ("2011-03-06", None),  # Sundays trade
        ("2010-12-24", "christmas_shutdown"),
        ("2011-01-03", "christmas_shutdown"),
        ("2011-01-04", None),
        ("2011-04-22", "bank_holiday"),  # Good Friday
        ("2011-04-24", "easter_sunday"),
        ("2011-04-29", "bank_holiday"),  # one-off royal wedding holiday
        ("2011-08-29", "bank_holiday"),
        ("2011-12-09", None),
    ],
)
def test_closure_rules(day, reason):
    assert closure_reason(day) == reason


def test_expected_trading_days_skip_closures():
    days = expected_trading_days("2011-12-09", 7)
    assert [d.day for d in days] == [11, 12, 13, 14, 15, 16, 18]  # Saturdays 10th, 17th skipped


def test_build_calendar_fields():
    cal = build_calendar("2011-04-22", "2011-04-26").set_index("date")
    assert cal["is_expected_trading_day"].tolist() == [False, False, False, False, True]
    assert cal.loc["2011-04-26", "day_of_week"] == 1


def test_compare_with_observed_reports_disagreements():
    observed = pd.Series(pd.to_datetime(["2011-03-04", "2011-03-05", "2011-03-07"]))
    result = compare_with_observed(observed)
    assert result["traded_on_expected_closed_day"] == ["2011-03-05"]
    assert result["no_sales_on_expected_trading_day"] == ["2011-03-06"]
