"""M5 benchmark adapter and metric tests on a tiny zip in the M5 file layout."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import m5
from src.forecasting import features as F
from src.forecasting import train as T

DAYS = 340  # d_331 is 2011-12-25
SPLIT = 300  # train-evaluation file holds d_1..d_300, test-evaluation d_301..d_340


def write_zip(path, sales: dict[str, np.ndarray], prices: dict[str, float], files=None):
    dates = pd.date_range(m5.M5_START, periods=DAYS)
    week = 11101 + (np.arange(DAYS) // 7)
    cal = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": week})

    def sales_frame(lo, hi):
        rows = []
        for item, units in sales.items():
            rows.append(
                {"item_id": item, "dept_id": "D", "cat_id": "C", "store_id": "CA_1",
                 "state_id": "CA", **{f"d_{d}": int(units[d - 1]) for d in range(lo, hi + 1)}}
            )  # fmt: skip
        other = {**rows[0], "item_id": "OTHER_STORE_ITEM", "store_id": "TX_1"}
        return pd.DataFrame([*rows, other])

    price_rows = [
        {"store_id": "CA_1", "item_id": item, "wm_yr_wk": w, "sell_price": p + (w - 11101) * 0.01}
        for item, p in prices.items()
        for w in np.unique(week)
    ]
    content = {
        "calendar.csv": cal,
        "sales_train_evaluation.csv": sales_frame(1, SPLIT),
        "sales_test_evaluation.csv": sales_frame(SPLIT + 1, DAYS),
        "sell_prices.csv": pd.DataFrame(price_rows),
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, df in content.items():
            if files is None or name in files:
                z.writestr(name, df.to_csv(index=False))
    return path


@pytest.fixture
def m5_zip(tmp_path):
    rng = np.random.default_rng(7)
    late = np.zeros(DAYS)
    late[100:] = rng.poisson(2.0, DAYS - 100)
    late[100] = 3  # first sale on d_101
    sales = {
        "ITEM_B": rng.poisson(5.0, DAYS).astype(float),
        "ITEM_A": rng.poisson(1.0, DAYS).astype(float),
        "ITEM_LATE": late,
    }
    sales["ITEM_B"][330] = 0  # Christmas (d_331): store closed
    sales["ITEM_A"][330] = 0
    sales["ITEM_LATE"][330] = 0
    sales["ITEM_A"][0] = 1  # make sure every item has a first sale on a known day
    sales["ITEM_B"][0] = 4
    return write_zip(tmp_path / "m5.zip", sales, {"ITEM_A": 2.0, "ITEM_B": 1.0, "ITEM_LATE": 3.0})


def test_load_store_shape_launch_and_christmas(m5_zip):
    daily = m5.load_store(m5_zip, "CA_1")
    assert set(daily["stock_code"]) == {"ITEM_A", "ITEM_B", "ITEM_LATE"}  # other store excluded
    expected = {"date", "stock_code", "units_sold", "gross_revenue", "units_returned"}
    assert set(daily.columns) >= expected | {"invoices", "customers", "is_partial_day"}
    # Train and test files are joined: the last day is d_340.
    assert daily["date"].max() == m5.M5_START + pd.Timedelta(days=DAYS - 1)
    # Items enter on their first sale, not with leading zeros.
    late = daily.loc[daily["stock_code"] == "ITEM_LATE"]
    assert late["date"].min() == m5.d_to_date(101)
    assert late["units_sold"].iloc[0] == 3
    # Christmas Day is not a trading day.
    assert not (daily["date"].dt.strftime("%m-%d") == "12-25").any()
    # No duplicated (item, day).
    assert not daily.duplicated(["stock_code", "date"]).any()


def test_revenue_uses_that_weeks_price(m5_zip):
    daily = m5.load_store(m5_zip, "CA_1").set_index(["stock_code", "date"])
    d = 50  # week index (50 - 1) // 7 = 7
    row = daily.loc[("ITEM_A", m5.d_to_date(d))]
    assert row["gross_revenue"] == pytest.approx(row["units_sold"] * (2.0 + 7 * 0.01))
    assert daily["invoices"].isna().all() and (daily["units_returned"] == 0).all()


def test_missing_evaluation_labels_is_a_clear_error(tmp_path):
    path = write_zip(
        tmp_path / "partial.zip",
        {"ITEM_A": np.ones(DAYS)},
        {"ITEM_A": 1.0},
        files={"calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"},
    )
    with pytest.raises(FileNotFoundError, match="sales_test_evaluation.csv"):
        m5.load_store(path, "CA_1")


def test_unknown_store_raises(m5_zip):
    with pytest.raises(ValueError, match="No M5 items"):
        m5.load_store(m5_zip, "WI_9")


def test_rmsse_matches_hand_computation():
    dates = pd.date_range("2020-01-01", periods=6)
    daily = pd.DataFrame(
        {
            "stock_code": ["X"] * 6,
            "date": dates,
            "units_sold": [2.0, 4.0, 2.0, 4.0, 3.0, 5.0],
            "gross_revenue": 1.0,
            "units_returned": 0.0,
            "invoices": np.nan,
            "customers": np.nan,
            "is_partial_day": False,
        }
    )
    panel = F.build_panel(daily)
    # Two forecasts made at day 3 (index 3) for days 4 and 5: actuals 3 and 5.
    rows = T.Rows(np.array([3, 3]), np.array([0, 0]), np.array([1, 2]), np.array(["model"] * 2))
    preds = {"flat": np.array([4.0, 4.0])}
    out = m5.rmsse(panel, rows, preds, dates[3], pd.Series({"X": 1.0}))
    # Scale: 1-step naive MSE over days 0..3 = mean(4, 4, 4) = 4. Forecast MSE = (1 + 1)/2 = 1.
    assert out["flat"]["rmsse_mean"] == pytest.approx(np.sqrt(1 / 4))
    assert out["flat"]["rmsse_revenue_weighted"] == pytest.approx(np.sqrt(1 / 4))


def test_benchmark_runs_end_to_end_on_tiny_data(m5_zip):
    daily = m5.load_store(m5_zip, "CA_1")
    report = m5.run(
        daily,
        train_end=m5.d_to_date(286),
        val_end=m5.d_to_date(300),
        test_end=m5.d_to_date(DAYS),
        train_origin_days=200,
        configs={"poisson": T.CONFIGS["poisson"]},
        max_rounds=20,
    )
    assert report["selected_config"] == "poisson"
    assert report["splits"]["test"]["targets"] == ["2011-11-25", "2012-01-03"]
    methods = report["test"]["segments"]["all"]["methods"]
    assert {"zero", "seasonal_naive", "moving_average_28", "pipeline"} <= set(methods)
    assert methods["zero"]["wape"] == 1.0
    assert set(report["test_rmsse"]) >= {"pipeline", "moving_average_28"}
    assert report["dataset"]["items"] == 3
    md = m5.render_markdown({**report, "dataset": {"store": "CA_1", **report["dataset"]}})
    assert "not comparable with the Kaggle leaderboard" in md
