"""Forecasting tests on small synthetic demand panels (test inputs, not real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import features as F
from src.forecasting import train as T
from src.forecasting.metrics import bias, mae, rmse, wape
from src.forecasting.predict import FALLBACK_METHOD, MODEL_METHOD, ForecastError, forecast
from src.processing.trading_calendar import expected_trading_days


def synthetic_daily(start="2010-01-04", days=560, seed=0, late_launch_at=None) -> pd.DataFrame:
    """Three products on the real trading calendar.

    STEADY sells every day with a weekday pattern; LUMPY sells on ~20% of days; LATE launches
    ``late_launch_at`` trading days before the end, so it has too little history.
    """
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp(start)] + expected_trading_days(start, days - 1)
    rows = []
    late_from = days - (late_launch_at or 5)
    for i, d in enumerate(dates):
        rows.append(("STEADY", d, int(20 + 5 * d.day_of_week + rng.integers(0, 5))))
        rows.append(("LUMPY", d, int(rng.integers(5, 50)) if rng.random() < 0.2 or i == 0 else 0))
        if i >= late_from:
            rows.append(("LATE", d, 3))
    df = pd.DataFrame(rows, columns=["stock_code", "date", "units_sold"])
    df["invoices"] = (df["units_sold"] > 0).astype(int) * 2
    df["customers"] = (df["units_sold"] > 0).astype(int)
    df["gross_revenue"] = df["units_sold"] * 2.0
    df["units_returned"] = 0
    df["is_partial_day"] = False
    return df


# --- metrics -----------------------------------------------------------------------------


def test_metrics_known_values():
    y, yhat = np.array([0, 10, 10]), np.array([1, 8, 13])
    assert mae(y, yhat) == pytest.approx(2.0)
    assert rmse(y, yhat) == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
    assert wape(y, yhat) == pytest.approx(6 / 20)
    assert bias(y, yhat) == pytest.approx(2 / 20)


def test_wape_is_nan_without_demand():
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


# --- features ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def daily():
    return synthetic_daily()


@pytest.fixture(scope="module")
def panel(daily):
    return F.build_panel(daily)


def test_origin_features_ignore_the_future(daily, panel):
    """Changing every value after day k must not change any origin feature at t <= k."""
    k = 300
    cutoff = panel.dates[k]
    tampered = daily.copy()
    future = tampered["date"] > cutoff
    tampered.loc[future, "units_sold"] = tampered.loc[future, "units_sold"] * 7 + 100
    tampered.loc[future, "gross_revenue"] = 1e6
    other = F.build_panel(tampered)
    for name in F.ORIGIN_FEATURES:
        np.testing.assert_array_equal(
            panel.origin[name][: k + 1], other.origin[name][: k + 1], err_msg=name
        )
    for name in F.STORE_FEATURES:
        np.testing.assert_array_equal(panel.store[name][: k + 1], other.store[name][: k + 1])
    np.testing.assert_array_equal(panel.eligible[: k + 1], other.eligible[: k + 1])


def test_feature_frame_ignores_the_future_for_every_horizon(daily, panel):
    k = 400
    tampered = daily.copy()
    tampered.loc[tampered["date"] > panel.dates[k], "units_sold"] = 10_000
    other = F.build_panel(tampered)
    # Every product x every horizon, so no case is silently all-NaN.
    n = len(panel.codes)
    t = np.full(7 * n, k)
    p = np.tile(np.arange(n), 7)
    h = np.repeat(np.arange(1, 8), n)
    assert np.isfinite(F.feature_frame(panel, t, p, h, panel.dates[t + h])["mean_7"]).any()
    td = panel.dates[t + h]
    pd.testing.assert_frame_equal(
        F.feature_frame(panel, t, p, h, td), F.feature_frame(other, t, p, h, td)
    )


def test_panel_marks_pre_launch_as_missing_and_gates_eligibility():
    panel = F.build_panel(synthetic_daily(late_launch_at=40))
    late = list(panel.codes).index("LATE")
    first = int(np.argmax(panel.units["LATE"].notna().to_numpy()))
    assert panel.units["LATE"].iloc[:first].isna().all()
    assert not panel.eligible[first, late]
    # Eligible once the window reaches MIN_WINDOW_DAYS with >= MIN_SELLING_DAYS selling days.
    assert panel.eligible[first + F.MIN_WINDOW_DAYS - 1, late]
    assert not panel.eligible[first + F.MIN_WINDOW_DAYS - 2, late]


def test_partial_days_are_excluded():
    daily = synthetic_daily()
    last = daily["date"].max()
    daily.loc[daily["date"] == last, "is_partial_day"] = True
    assert F.build_panel(daily).dates[-1] < last


def test_seasonal_naive_uses_same_weekday_at_or_before_origin(panel):
    steady = list(panel.codes).index("STEADY")
    t = 200
    for h in range(1, 8):
        target = panel.dates[t + h]
        got = F.seasonal_naive(panel, np.array([t]), np.array([steady]), [target])[0]
        candidates = [
            i for i in range(t, t - 8, -1) if panel.dates[i].day_of_week == target.day_of_week
        ]
        assert got == panel.units["STEADY"].iloc[candidates[0]]


def test_last_year_feature_uses_dates_a_year_back(panel):
    target = panel.dates[-1]
    ly = F.last_year_mean(panel, pd.DatetimeIndex([target]))[0]
    steady = list(panel.codes).index("STEADY")
    window = panel.units["STEADY"][
        (panel.dates >= target - pd.Timedelta(days=367))
        & (panel.dates <= target - pd.Timedelta(days=361))
    ]
    assert ly[steady] == pytest.approx(window.mean())


# --- splits ------------------------------------------------------------------------------


def test_training_rows_respect_last_target_and_eligibility(panel):
    last_target = panel.dates[350]
    rows = T.training_rows(panel, last_target)
    assert (rows.t_idx + rows.horizon <= 350).all()
    assert set(np.unique(rows.horizon)) == set(F.HORIZONS)
    assert panel.eligible[rows.t_idx, rows.p_idx].all()
    again = T.training_rows(panel, last_target)
    np.testing.assert_array_equal(rows.horizon, again.horizon)


def test_evaluation_rows_cover_window_and_segments():
    panel = F.build_panel(synthetic_daily(late_launch_at=20))
    start, end = panel.dates[-30], panel.dates[-1]
    rows = T.evaluation_rows(panel, start, end)
    td = T.target_dates(panel, rows)
    assert (td >= start).all() and (td <= end).all()
    assert (rows.t_idx + rows.horizon < len(panel.dates)).all()
    segs = set(rows.segment)
    assert {"model", "fallback", "launched_after_origin"} <= segs
    # Every (target, horizon) pair appears once per product.
    keys = pd.DataFrame({"t": rows.t_idx, "p": rows.p_idx, "h": rows.horizon})
    assert not keys.duplicated().any()


def test_chronological_splits_do_not_overlap():
    assert T.TRAIN_END < T.VAL_END
    panel = F.build_panel(synthetic_daily(start="2010-01-04", days=600))
    train = T.training_rows(panel, T.TRAIN_END)
    val = T.evaluation_rows(panel, T.TRAIN_END + pd.Timedelta(days=1), T.VAL_END)
    assert T.target_dates(panel, train).max() <= T.TRAIN_END
    assert T.target_dates(panel, val).min() > T.TRAIN_END


# --- end to end --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(monkeypatch_module):
    monkeypatch_module.setattr(T, "MAX_ROUNDS", 60)
    monkeypatch_module.setattr(T, "CONFIGS", {"tweedie_1.2": T.CONFIGS["tweedie_1.2"]})
    monkeypatch_module.setitem(T.BASE_PARAMS, "min_data_in_leaf", 20)
    daily = synthetic_daily(start="2010-01-04", days=600, late_launch_at=15)
    result = T.run(daily)
    return daily, result


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_run_reports_all_methods_on_test(trained):
    _, result = trained
    model_seg = result["report"]["test"]["segments"]["model"]
    assert {"zero", "seasonal_naive", "moving_average_28", "lightgbm", "pipeline"} <= set(
        model_seg["methods"]
    )
    for m in model_seg["methods"].values():
        assert {"mae", "rmse", "wape", "bias", "by_horizon"} <= set(m)
    assert result["meta"]["features"] == F.FEATURES
    assert result["backtest"]["target_date"].min() > T.VAL_END


def test_model_roundtrip_and_reproducible_forecast(trained, tmp_path):
    daily, result = trained
    model_path, meta = tmp_path / "m.txt", result["meta"]
    result["booster"].save_model(str(model_path))
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_path))
    a = forecast(daily, booster, meta)
    b = forecast(daily, booster, meta)
    pd.testing.assert_frame_equal(a, b)
    # The reloaded model gives the same numbers as the in-memory one.
    c = forecast(daily, result["booster"], meta)
    np.testing.assert_allclose(a["forecast_units"], c["forecast_units"])


def test_forecast_rows_methods_and_calendar(trained):
    daily, result = trained
    fc = forecast(daily, result["booster"], result["meta"])
    origin = daily["date"].max()
    assert set(fc["horizon_trading_days"]) == set(range(1, 8))
    assert (fc["forecast_units"] >= 0).all()
    assert (fc["target_date"].dt.day_of_week != 5).all()  # never a Saturday
    assert list(sorted(fc["target_date"].unique())) == expected_trading_days(origin, 7)
    methods = fc.drop_duplicates("stock_code").set_index("stock_code")["method"]
    assert methods["STEADY"] == MODEL_METHOD
    assert methods["LATE"] == FALLBACK_METHOD
    late = fc[fc["stock_code"] == "LATE"]
    assert (late["forecast_units"] == 3.0).all()  # constant 3/day, launch day excluded
    assert set(fc["value_type"]) == {"model_prediction"}


def test_forecast_at_past_origin_ignores_later_data(trained):
    daily, result = trained
    origin = sorted(daily["date"].unique())[-20]
    full = forecast(daily, result["booster"], result["meta"], origin=origin)
    truncated = forecast(
        daily[daily["date"] <= origin], result["booster"], result["meta"], origin=origin
    )
    pd.testing.assert_frame_equal(full, truncated)


def test_forecast_rejects_bad_origin(trained):
    daily, result = trained
    with pytest.raises(ForecastError):
        forecast(daily, result["booster"], result["meta"], origin="2010-01-09")  # a Saturday


def test_fallback_excludes_launch_day():
    daily = synthetic_daily(late_launch_at=6)
    daily.loc[daily["stock_code"] == "LATE", "units_sold"] = [50, 2, 2, 4, 0, 2]
    panel = F.build_panel(daily)
    late = list(panel.codes).index("LATE")
    t = len(panel.dates) - 1
    got = F.fallback_forecast(panel, np.array([t, t - 5]), np.array([late, late]))
    # Six days observed: drop the launch day (50) and average the other five.
    assert got[0] == pytest.approx((2 + 2 + 4 + 0 + 2) / 5)
    # Only the launch day observed: nothing left to average.
    assert got[1] == 0.0


def test_cumulative_7day_uses_complete_origins_only():
    bt = pd.DataFrame(
        {
            "stock_code": ["A"] * 7 + ["B"] * 3,
            "origin_date": [pd.Timestamp("2011-10-03")] * 10,
            "horizon": list(range(1, 8)) + [1, 2, 3],
            "segment": ["model"] * 10,
            "actual_units": [1.0] * 10,
            "forecast_zero": [0.0] * 10,
            "forecast_lightgbm": [2.0] * 10,
        }
    )
    res = T.cumulative_7day(bt)
    assert res["model"]["lightgbm"]["rows"] == 1
    assert res["model"]["lightgbm"]["mae"] == pytest.approx(7.0)
    assert res["model"]["zero"]["wape"] == pytest.approx(1.0)
