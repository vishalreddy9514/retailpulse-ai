"""Leakage-safe demand features for product-level daily forecasting.

Everything is computed on a ``(trading day x product)`` matrix of units sold. Every
"origin" feature at row ``t`` uses only rows ``<= t``: pandas rolling and expanding
windows end at the current row, and lags shift backwards. A forecast made at the end
of day ``t`` can therefore read row ``t`` and nothing later. ``tests/unit/test_forecasting.py``
checks this by changing the future and asserting that the features do not move.

Days before a product's first sale are NaN, not zero, so a product is not penalised for
the time before it existed. Horizons are counted in *trading* days.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A product gets a model forecast only once it has this much history at the origin.
MIN_SELLING_DAYS = 10
MIN_WINDOW_DAYS = 28
HORIZONS = tuple(range(1, 8))

ORIGIN_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_5",
    "lag_6",
    "lag_7",
    "mean_7",
    "mean_14",
    "mean_28",
    "mean_56",
    "mean_112",
    "std_28",
    "max_28",
    "nonzero_share_28",
    "nonzero_share_112",
    "mean_nonzero_56",
    "days_since_last_sale",
    "age_days",
    "selling_days_to_date",
    "expanding_mean",
    "invoices_mean_28",
    "customers_mean_28",
    "price_28",
    "price_ratio_28_112",
    "return_rate_28",
]
STORE_FEATURES = ["store_mean_7", "store_mean_28", "store_ratio_7_28"]
TARGET_FEATURES = [
    "horizon",
    "target_day_of_week",
    "target_week_of_year",
    "target_month",
    "target_day_of_month",
    "last_year_mean_7",
]
FEATURES = ORIGIN_FEATURES + STORE_FEATURES + TARGET_FEATURES


@dataclass
class Panel:
    dates: pd.DatetimeIndex  # trading days, ascending
    codes: np.ndarray  # product codes, column order
    units: pd.DataFrame  # (T, P) units sold, NaN before first sale
    origin: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (T, P) float32
    store: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (T,) float32
    eligible: np.ndarray | None = None  # (T, P) bool: model-eligible at origin t
    first_row: np.ndarray | None = None  # (P,) row of each product's first sale

    def index_of(self, day: pd.Timestamp) -> int:
        pos = self.dates.get_indexer([pd.Timestamp(day)])[0]
        if pos < 0:
            raise KeyError(f"{day} is not a trading day in the panel")
        return int(pos)


def _wide(daily: pd.DataFrame, column: str, codes, dates) -> pd.DataFrame:
    wide = daily.pivot(index="date", columns="stock_code", values=column)
    return wide.reindex(index=dates, columns=codes).astype("float64")


def build_panel(product_daily: pd.DataFrame, exclude_partial: bool = True) -> Panel:
    """Build the feature matrices from ``product_daily`` (output of aggregate_sales)."""
    daily = product_daily
    if exclude_partial and "is_partial_day" in daily:
        # A truncated day would look like a demand collapse; it is neither a feature nor a target.
        daily = daily.loc[~daily["is_partial_day"]]
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    codes = np.array(sorted(daily["stock_code"].unique()), dtype=object)

    units = _wide(daily, "units_sold", codes, dates)
    panel = Panel(dates=dates, codes=codes, units=units)

    def roll(df: pd.DataFrame, w: int) -> pd.core.window.Rolling:
        return df.rolling(w, min_periods=1)

    sold = (units > 0).astype("float64").where(units.notna())
    o: dict[str, pd.DataFrame] = {}
    for k in range(1, 8):
        o[f"lag_{k}"] = units.shift(k - 1)
    for w in (7, 14, 28, 56, 112):
        o[f"mean_{w}"] = roll(units, w).mean()
    o["std_28"] = roll(units, 28).std(ddof=0)
    o["max_28"] = roll(units, 28).max()
    o["nonzero_share_28"] = roll(sold, 28).mean()
    o["nonzero_share_112"] = roll(sold, 112).mean()
    nz56 = roll(sold, 56).sum()
    o["mean_nonzero_56"] = roll(units, 56).sum() / nz56.where(nz56 > 0)

    row = np.arange(len(dates), dtype="float64")[:, None]
    last_sale_row = pd.DataFrame(np.where(units.gt(0), row, np.nan), index=dates).ffill()
    o["days_since_last_sale"] = pd.DataFrame(row - last_sale_row.to_numpy(), index=dates)
    first_row = units.notna().idxmax().map(dates.get_loc).to_numpy(dtype="float64")
    panel.first_row = first_row.astype("int64")
    age = row - first_row[None, :]
    o["age_days"] = pd.DataFrame(np.where(age >= 0, age, np.nan), index=dates)
    o["selling_days_to_date"] = sold.cumsum()
    o["expanding_mean"] = units.expanding(min_periods=1).mean()

    invoices = _wide(daily, "invoices", codes, dates)
    customers = _wide(daily, "customers", codes, dates)
    revenue = _wide(daily, "gross_revenue", codes, dates)
    returned = _wide(daily, "units_returned", codes, dates)
    o["invoices_mean_28"] = roll(invoices, 28).mean()
    o["customers_mean_28"] = roll(customers, 28).mean()
    units28 = roll(units, 28).sum()
    units112 = roll(units, 112).sum()
    price28 = roll(revenue, 28).sum() / units28.where(units28 > 0)
    price112 = roll(revenue, 112).sum() / units112.where(units112 > 0)
    o["price_28"] = price28
    o["price_ratio_28_112"] = price28 / price112
    o["return_rate_28"] = roll(returned, 28).sum() / units28.where(units28 > 0)

    panel.origin = {k: np.asarray(v, dtype="float32") for k, v in o.items()}

    store = units.sum(axis=1, min_count=1)
    s7 = store.rolling(7, min_periods=1).mean()
    s28 = store.rolling(28, min_periods=1).mean()
    panel.store = {
        "store_mean_7": s7.to_numpy("float32"),
        "store_mean_28": s28.to_numpy("float32"),
        "store_ratio_7_28": (s7 / s28).to_numpy("float32"),
    }

    window = o["age_days"].to_numpy() + 1
    selling = o["selling_days_to_date"].to_numpy()
    panel.eligible = (window >= MIN_WINDOW_DAYS) & (selling >= MIN_SELLING_DAYS)
    return panel


def last_year_mean(panel: Panel, target_dates: pd.DatetimeIndex) -> np.ndarray:
    """Mean units per product over the 7 calendar days centred 364 days before each target.

    364 days keeps the weekday aligned. The window ends a year before the target, so it is
    always in the past at any origin up to 7 trading days earlier. NaN when the product did
    not exist then. Returns shape (len(target_dates), P).
    """
    values = panel.units.to_numpy()
    filled = np.nan_to_num(values, nan=0.0)
    cs = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(filled, axis=0)])
    cnt = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(~np.isnan(values), axis=0)])
    start = panel.dates.searchsorted(target_dates - pd.Timedelta(days=367), side="left")
    end = panel.dates.searchsorted(target_dates - pd.Timedelta(days=361), side="right")
    total = cs[end] - cs[start]
    count = cnt[end] - cnt[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / count, np.nan)
    return out.astype("float32")


def feature_frame(
    panel: Panel,
    t_idx: np.ndarray,
    p_idx: np.ndarray,
    horizon: np.ndarray,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Gather model features for rows (origin index, product index, horizon, target date)."""
    target_dates = pd.DatetimeIndex(target_dates)
    data: dict[str, np.ndarray] = {}
    for name in ORIGIN_FEATURES:
        data[name] = panel.origin[name][t_idx, p_idx]
    for name in STORE_FEATURES:
        data[name] = panel.store[name][t_idx]

    unique_targets, inverse = np.unique(target_dates.to_numpy(), return_inverse=True)
    ly = last_year_mean(panel, pd.DatetimeIndex(unique_targets))
    data["horizon"] = horizon.astype("float32")
    data["target_day_of_week"] = target_dates.day_of_week.to_numpy("float32")
    data["target_week_of_year"] = target_dates.isocalendar().week.to_numpy("float32")
    data["target_month"] = target_dates.month.to_numpy("float32")
    data["target_day_of_month"] = target_dates.day.to_numpy("float32")
    data["last_year_mean_7"] = ly[inverse, p_idx]
    return pd.DataFrame(data, columns=FEATURES)


def seasonal_naive(panel: Panel, t_idx: np.ndarray, p_idx: np.ndarray, target_dates) -> np.ndarray:
    """Units on the most recent trading day at or before the origin with the target's weekday."""
    target_dow = pd.DatetimeIndex(target_dates).day_of_week.to_numpy()
    dows = panel.dates.day_of_week.to_numpy()
    values = panel.units.to_numpy()
    out = np.full(len(t_idx), np.nan)
    # The matching day is at most 7 trading days back (6 trading days per week, one may be shut).
    for back in range(0, 8):
        rows = t_idx - back
        ok = np.isnan(out) & (rows >= 0)
        hit = ok & (dows[np.clip(rows, 0, None)] == target_dow)
        out[hit] = values[rows[hit], p_idx[hit]]
    return np.nan_to_num(out, nan=0.0)


def moving_average(panel: Panel, t_idx: np.ndarray, p_idx: np.ndarray, window: int = 28):
    name = f"mean_{window}"
    return np.nan_to_num(panel.origin[name][t_idx, p_idx].astype("float64"), nan=0.0)


def fallback_forecast(panel: Panel, t_idx: np.ndarray, p_idx: np.ndarray) -> np.ndarray:
    """Forecast for products with too little history: 28-day mean excluding the launch day.

    A product enters the data on the day of its first sale, so its first observed day always
    has demand. A plain average over the few days since launch is therefore biased upward
    (+11% on validation). Dropping the launch day while it is inside the window removes
    that bias (-0.07% on validation). With only the launch day observed, the forecast is 0.
    """
    window = 28
    age = panel.origin["age_days"][t_idx, p_idx].astype("float64") + 1
    n = np.minimum(age, window)
    total = moving_average(panel, t_idx, p_idx, window) * n
    launch = np.nan_to_num(panel.units.to_numpy()[panel.first_row[p_idx], p_idx], nan=0.0)
    in_window = age <= window
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(in_window, (total - launch) / (n - 1), total / n)
    out = np.where(in_window & (n <= 1), 0.0, out)
    return np.nan_to_num(np.clip(out, 0, None), nan=0.0)
