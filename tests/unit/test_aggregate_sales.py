from __future__ import annotations

import pandas as pd
import pytest

from src.processing.aggregate_sales import aggregate, classify_demand
from src.processing.clean_transactions import clean_transactions


@pytest.fixture
def tables(make_raw):
    rows = [
        # Product 11111 trades from Mon 7 Mar; nothing on Tue 8 Mar.
        ("500001", "11111", "MUG", 10, "2011-03-07 10:00", 2.0, 1.0, "UK"),
        ("500002", "11111", "MUG", 5, "2011-03-07 11:00", 2.0, 2.0, "UK"),
        ("500003", "11111", "MUG", 4, "2011-03-09 10:00", 2.5, None, "UK"),
        # Return on a later day: tracked separately, demand unchanged.
        ("C500004", "11111", "MUG", -3, "2011-03-10 09:00", 2.0, 1.0, "UK"),
        # Product 22222 first sells on Tue 8 Mar.
        ("500005", "22222", "BOWL", 1, "2011-03-08 10:00", 5.0, 3.0, "UK"),
        # Same-day reversal on 22222: excluded from demand, kept in units_reversed.
        ("500006", "22222", "BOWL", 100, "2011-03-09 10:00", 5.0, 4.0, "UK"),
        ("C500007", "22222", "BOWL", -100, "2011-03-09 10:05", 5.0, 4.0, "UK"),
        # Late sale so the last day is a full day, plus postage that never counts.
        ("500008", "22222", "BOWL", 2, "2011-03-10 17:00", 5.0, 3.0, "UK"),
        ("500009", "POST", "POSTAGE", 1, "2011-03-10 17:00", 18.0, 3.0, "UK"),
    ]
    clean, _ = clean_transactions(make_raw(rows))
    out, report = aggregate(clean)
    return out, report


def _row(daily: pd.DataFrame, code: str, day: str) -> pd.Series:
    return daily.set_index(["stock_code", "date"]).loc[(code, pd.Timestamp(day))]


def test_dense_grid_starts_at_first_sale_with_explicit_zeros(tables):
    out, report = tables
    daily = out["product_daily"]
    assert sorted(daily.loc[daily.stock_code == "11111", "date"].dt.day) == [7, 8, 9, 10]
    # 22222 has no rows before its first sale.
    assert sorted(daily.loc[daily.stock_code == "22222", "date"].dt.day) == [8, 9, 10]
    assert _row(daily, "11111", "2011-03-08")["units_sold"] == 0
    assert report.product_day_rows == 7
    assert "POST" not in set(daily["stock_code"])


def test_daily_metrics(tables):
    daily = tables[0]["product_daily"]
    r = _row(daily, "11111", "2011-03-07")
    assert (r["units_sold"], r["sale_lines"], r["invoices"], r["customers"]) == (15, 2, 2, 2)
    assert r["gross_revenue"] == 30.0
    assert r["avg_unit_price"] == 2.0
    # The missing-customer sale still counts as demand.
    assert _row(daily, "11111", "2011-03-09")["units_sold"] == 4
    assert _row(daily, "11111", "2011-03-09")["customers"] == 0


def test_returns_are_separate_from_demand(tables):
    r = _row(tables[0]["product_daily"], "11111", "2011-03-10")
    assert r["units_sold"] == 0
    assert r["units_returned"] == 3
    assert r["return_value"] == 6.0
    assert r["net_units"] == -3
    assert r["net_revenue"] == -6.0


def test_same_day_reversal_not_counted_as_demand(tables):
    r = _row(tables[0]["product_daily"], "22222", "2011-03-09")
    assert r["units_sold"] == 0
    assert r["units_reversed"] == 100
    assert r["units_returned"] == 0


def test_reconciliation_checks_pass(tables):
    _, report = tables
    assert report.passed, report.reconciliation


def test_daily_totals_and_summary(tables):
    out, _ = tables
    totals = out["daily_totals"].set_index("date")
    assert totals.loc["2011-03-07", "units_sold"] == 15
    assert totals["units_sold"].sum() == out["product_daily"]["units_sold"].sum()
    summary = out["product_summary"].set_index("stock_code")
    assert summary.loc["11111", "selling_days"] == 2
    assert summary.loc["11111", "window_days"] == 4
    assert summary.loc["11111", "adi"] == 2.0
    assert summary.loc["11111", "history_status"] == "insufficient"
    assert summary.loc["11111", "product_description"] == "MUG"


def test_partial_last_day_is_flagged(make_raw):
    rows = [
        (f"5000{i:02d}", "11111", "MUG", 1, f"2011-03-{d:02d} {h}", 1.0, 1.0, "UK")
        for i, (d, h) in enumerate(
            [
                (1, "09:00"),
                (1, "18:00"),
                (2, "09:00"),
                (2, "18:00"),
                (3, "09:00"),
                (3, "18:00"),
                (4, "09:00"),
                (4, "18:00"),
                (6, "09:00"),
                (6, "18:00"),
                (7, "09:00"),
                (7, "18:00"),
                (8, "09:00"),
                (8, "18:00"),
                (9, "09:00"),
                (9, "18:00"),
                (10, "09:00"),
                (10, "18:00"),
                (11, "09:00"),
                (11, "18:00"),
                (13, "09:00"),
                (13, "18:00"),
                (14, "09:00"),
            ]
        )
    ]
    clean, _ = clean_transactions(make_raw(rows))
    out, report = aggregate(clean)
    assert report.partial_days == ["2011-03-14"]
    assert out["daily_totals"].set_index("date").loc["2011-03-14", "is_partial_day"]


def test_classify_demand_syntetos_boylan():
    adi = pd.Series([1.0, 1.0, 2.0, 2.0])
    cv2 = pd.Series([0.1, 1.0, 0.1, 1.0])
    assert classify_demand(adi, cv2).tolist() == ["smooth", "erratic", "intermittent", "lumpy"]
