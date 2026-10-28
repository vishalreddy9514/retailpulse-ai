"""Optional benchmark: the RetailPulse forecasting pipeline on the Walmart M5 dataset.

This is a second, independent dataset for checking that the UCI pipeline generalises. It
is not used by the product (no M5 data reaches the API, copilot or dashboard).

Data: the M5 Forecasting competition files (Makridakis, Spiliotis & Assimakopoulos, 2022,
"M5 accuracy competition: Results, findings, and conclusions", IJF 38(4)). Originally on
Kaggle under the competition rules; this module reads the zip mirrored by Nixtla
(github.com/Nixtla/m5-forecasts), which also contains the released labels for the final
evaluation period. The zip is downloaded to ``data/raw/m5/`` and never committed.

What runs is the UCI pipeline unchanged: same features, same model configs, same
rolling-origin evaluation over horizons 1-7, same segments and baselines. Only the adapter
and the split dates are M5-specific:

* one store (default ``CA_1``, 3,049 items), to stay within a laptop's memory and time
* items enter on their first sale (M5 has leading zeros before an item is stocked)
* Christmas Day is dropped as a non-trading day (every store sells nothing)
* ``invoices`` and ``customers`` do not exist in M5 and are left missing; returns are 0
* revenue is units x weekly ``sell_price``
* M5 extras (SNAP days, events) are deliberately *not* added: the point is to measure the
  pipeline as built, not to tune it for M5

Splits follow M5's own periods:

* train:      targets up to d_1913 (2016-04-24), origins from the last ``train_origin_days``
* validation: d_1914-d_1941 (2016-04-25 to 2016-05-22), M5's public-leaderboard period
* test:       d_1942-d_1969 (2016-05-23 to 2016-06-19), M5's final evaluation period

Besides MAE/RMSE/WAPE this reports item-level RMSSE (M5's scale-free error), unweighted and
weighted by each item's revenue over the 28 days before the test. It is NOT the
competition's WRMSSE: that averages 12 aggregation levels and scores one 28-day-ahead
forecast, while this scores rolling 1-7 day forecasts for one store's items. Do not
compare the numbers with the Kaggle leaderboard.

Usage::

    python -m src.benchmarks.m5 --download        # fetch the zip (~50 MB) and run
    python -m src.benchmarks.m5 --store TX_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.forecasting import features as F
from src.forecasting import train as T

logger = logging.getLogger(__name__)

MIRROR_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
MIRROR_SHA256 = "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"
ZIP_PATH = config.RAW_DIR / "m5" / "m5.zip"
REPORT_JSON = config.REPORTS_DIR / "m5_benchmark.json"
REPORT_MD = config.REPORTS_DIR / "m5_benchmark.md"

M5_START = pd.Timestamp("2011-01-29")  # d_1
TRAIN_END_D, VAL_END_D, TEST_END_D = 1913, 1941, 1969
SALES_FILES = ("sales_train_evaluation.csv", "sales_test_evaluation.csv")


def d_to_date(d: int) -> pd.Timestamp:
    return M5_START + pd.Timedelta(days=d - 1)


TRAIN_END, VAL_END, TEST_END = (d_to_date(d) for d in (TRAIN_END_D, VAL_END_D, TEST_END_D))


def download(dest: Path = ZIP_PATH, url: str = MIRROR_URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != MIRROR_SHA256:
        tmp.unlink()
        raise ValueError(f"M5 zip checksum mismatch: {digest}")
    tmp.replace(dest)
    return dest


def load_store(zip_path: Path | str, store: str = "CA_1", max_items: int | None = None):
    """M5 sales for one store in the ``product_daily`` shape the forecasting code expects."""
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        missing = [n for n in (*SALES_FILES, "sell_prices.csv") if n not in names]
        if missing:
            raise FileNotFoundError(f"{zip_path} lacks {missing}; use the M5 evaluation files")
        wide = pd.concat(
            [
                pd.read_csv(z.open(n)).query("store_id == @store").set_index("item_id")
                for n in SALES_FILES
            ],
            axis=1,
        ).filter(like="d_")
        prices = pd.read_csv(z.open("sell_prices.csv")).query("store_id == @store")
    if wide.empty:
        raise ValueError(f"No M5 items for store {store!r}")
    wide = wide.sort_index()
    if max_items is not None:
        wide = wide.iloc[:max_items]
    days = np.array([int(c[2:]) for c in wide.columns])
    if not (np.diff(days) == 1).all() or days[0] != 1:
        raise ValueError("M5 day columns are not contiguous from d_1")

    long = wide.stack().rename("units_sold").reset_index()
    long.columns = ["stock_code", "d", "units_sold"]
    long["d"] = long["d"].str[2:].astype("int32")
    long["date"] = M5_START + pd.to_timedelta(long["d"] - 1, unit="D")
    # Items enter on their first sale, as UCI products do; leading zeros mean "not stocked".
    first = long.loc[long["units_sold"] > 0].groupby("stock_code")["d"].min()
    long = long.loc[long["d"] >= long["stock_code"].map(first)]
    # Christmas Day: every store is closed.
    long = long.loc[long["date"].dt.strftime("%m-%d") != "12-25"]

    long["wm_yr_wk"] = long["d"].map(_week_codes(zip_path))
    long = long.merge(
        prices[["item_id", "wm_yr_wk", "sell_price"]].rename(columns={"item_id": "stock_code"}),
        on=["stock_code", "wm_yr_wk"],
        how="left",
    )
    units = long["units_sold"].astype("float64")
    return pd.DataFrame(
        {
            "stock_code": long["stock_code"].astype(str),
            "date": long["date"],
            "units_sold": units,
            "gross_revenue": units * long["sell_price"],
            "units_returned": 0.0,
            "invoices": np.nan,  # not in M5
            "customers": np.nan,  # not in M5
            "is_partial_day": False,
        }
    ).sort_values(["stock_code", "date"], ignore_index=True)


def _week_codes(zip_path: Path | str) -> pd.Series:
    """d -> wm_yr_wk from calendar.csv (row k is d_{k+1}; the mirror drops the ``d`` column)."""
    with zipfile.ZipFile(zip_path) as z:
        cal = pd.read_csv(z.open("calendar.csv"), parse_dates=["date"])
    if not (cal["date"] == pd.date_range(M5_START, periods=len(cal))).all():
        raise ValueError("calendar.csv is not one row per day from 2011-01-29")
    return pd.Series(cal["wm_yr_wk"].to_numpy(), index=np.arange(1, len(cal) + 1))


# -- metrics ---------------------------------------------------------------------------------


def rmsse(
    panel: F.Panel,
    rows: T.Rows,
    preds: dict[str, np.ndarray],
    scale_end: pd.Timestamp,
    weight_revenue: pd.Series,
) -> dict:
    """Item-level RMSSE: sqrt(MSE of the forecasts / MSE of the in-sample 1-step naive).

    The scale uses each item's history from its first sale to ``scale_end`` (the last day
    known when the test forecasts were made). Items whose history never changes have no
    scale and are excluded and counted.
    """
    hist = panel.units.loc[:scale_end].to_numpy()
    diffs = np.diff(hist, axis=0)
    with np.errstate(invalid="ignore"):
        scale = np.nanmean(diffs**2, axis=0)
    y = T.targets(panel, rows)
    codes = pd.Index(panel.codes)
    w_all = weight_revenue.reindex(codes).fillna(0.0).to_numpy()
    out = {}
    for name, yhat in preds.items():
        if np.isnan(yhat).any():  # lightgbm is only defined on the model segment
            continue
        sq = pd.Series((y - yhat) ** 2).groupby(rows.p_idx).mean()
        p = sq.index.to_numpy()
        ok = np.isfinite(scale[p]) & (scale[p] > 0)
        per_item = np.sqrt(sq.to_numpy()[ok] / scale[p][ok])
        w = w_all[p][ok]
        out[name] = {
            "items": int(ok.sum()),
            "items_without_scale": int((~ok).sum()),
            "rmsse_mean": round(float(per_item.mean()), 4),
            "rmsse_revenue_weighted": round(float((per_item * w).sum() / w.sum()), 4),
        }
    return out


# -- benchmark -------------------------------------------------------------------------------


def _from_origin(panel: F.Panel, rows: T.Rows, first_origin: pd.Timestamp) -> T.Rows:
    return rows.take(rows.t_idx >= panel.dates.searchsorted(first_origin))


def run(
    daily: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
    test_end: pd.Timestamp = TEST_END,
    train_origin_days: int = 730,
    configs: dict[str, dict] | None = None,
    max_rounds: int | None = None,
) -> dict:
    started = time.monotonic()
    configs = T.CONFIGS if configs is None else configs
    panel = F.build_panel(daily)
    day = pd.Timedelta(days=1)
    logger.info("M5 panel: %d days x %d items", len(panel.dates), len(panel.codes))

    # Training origins are limited to the last train_origin_days before each cut-off; the
    # features still use the full history (rolling windows, expanding means, last year).
    train_rows = _from_origin(
        panel, T.training_rows(panel, train_end), train_end - pd.Timedelta(days=train_origin_days)
    )
    val_rows = T.evaluation_rows(panel, train_end + day, val_end)
    val_model = val_rows.take(val_rows.segment == "model")
    x_train, y_train = T.design(panel, train_rows), T.targets(panel, train_rows)
    x_val, y_val = T.design(panel, val_model), T.targets(panel, val_model)
    dtrain = lgb.Dataset(x_train, y_train, free_raw_data=False)
    dval = lgb.Dataset(x_val, y_val, reference=dtrain, free_raw_data=False)
    logger.info("Training rows: %d, validation model rows: %d", len(train_rows), len(val_model))
    selection, best = T.select_config(dtrain, dval, x_val, y_val, configs, max_rounds)
    params = {**T.BASE_PARAMS, **configs[best]}
    rounds = selection[best]["best_iteration"]

    val_booster = T.fit(params, dtrain, rounds)
    validation = T.evaluate(panel, val_rows, T.predict_rows(panel, val_rows, val_booster))

    # Test: refit on train + validation with the selected config, score once.
    trval = _from_origin(
        panel, T.training_rows(panel, val_end), val_end - pd.Timedelta(days=train_origin_days)
    )
    booster = T.fit(params, lgb.Dataset(T.design(panel, trval), T.targets(panel, trval)), rounds)
    test_rows = T.evaluation_rows(panel, val_end + day, test_end)
    test_preds = T.predict_rows(panel, test_rows, booster)
    test = T.evaluate(panel, test_rows, test_preds)

    backtest = pd.DataFrame(
        {
            "stock_code": panel.codes[test_rows.p_idx],
            "origin_date": panel.dates[test_rows.t_idx],
            "horizon": test_rows.horizon,
            "segment": test_rows.segment.astype(str),
            "actual_units": T.targets(panel, test_rows),
            **{f"forecast_{k}": v for k, v in test_preds.items()},
        }
    )
    recent = daily.loc[
        (daily["date"] > val_end - pd.Timedelta(days=28)) & (daily["date"] <= val_end)
    ]
    weights = recent.groupby("stock_code")["gross_revenue"].sum()

    return {
        "dataset": {
            "name": "M5 Forecasting (Walmart), evaluation files",
            "source": MIRROR_URL,
            "sha256": MIRROR_SHA256,
            "items": int(len(panel.codes)),
            "days": int(len(panel.dates)),
            "first_day": str(panel.dates[0].date()),
            "last_day": str(panel.dates[-1].date()),
            "zero_share_after_first_sale": round(
                float((panel.units.to_numpy() == 0).sum() / panel.units.notna().to_numpy().sum()), 4
            ),  # fmt: skip
        },
        "splits": {
            "train": {
                "last_target": str(train_end.date()),
                "first_origin": str((train_end - pd.Timedelta(days=train_origin_days)).date()),
                "rows": len(train_rows),
            },
            "validation": {"targets": [str((train_end + day).date()), str(val_end.date())]},
            "test": {"targets": [str((val_end + day).date()), str(test_end.date())]},
        },
        "config_selection": selection,
        "selected_config": best,
        "validation": validation,
        "test": test,
        "test_cumulative_7day": T.cumulative_7day(backtest),
        "test_rmsse": rmsse(panel, test_rows, test_preds, val_end, weights),
        "runtime_seconds": round(time.monotonic() - started, 1),
    }


def render_markdown(report: dict) -> str:
    ds = report["dataset"]
    test = report["test"]["segments"]
    lines = [
        "# M5 benchmark (optional)\n",
        "Generated by `python -m src.benchmarks.m5`. The RetailPulse forecasting pipeline, "
        "unchanged, on Walmart M5 data for one store. Units: item units per day.",
        f"Store **{ds['store']}**: {ds['items']:,} items, {ds['days']:,} days "
        f"({ds['first_day']} to {ds['last_day']}), "
        f"{ds['zero_share_after_first_sale']:.1%} zero-sale days after first sale.",
        f"Test targets {report['splits']['test']['targets'][0]} to "
        f"{report['splits']['test']['targets'][1]} (M5 final evaluation period), "
        "rolling origins, horizons 1-7 days.",
        f"Selected config: **{report['selected_config']}** (validation WAPE "
        + ", ".join(
            f"{k} {v['validation_wape']:.4f}" for k, v in report["config_selection"].items()
        )
        + ").\n",
        "## Test, all items\n",
        "| method | MAE | RMSE | WAPE | bias | RMSSE (mean) | RMSSE (revenue-weighted) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rm = report["test_rmsse"]
    for name, m in test["all"]["methods"].items():
        r = rm.get(name)
        extra = f"{r['rmsse_mean']:.3f} | {r['rmsse_revenue_weighted']:.3f}" if r else "- | -"
        lines.append(
            f"| {name} | {m['mae']:.3f} | {m['rmse']:.2f} | {m['wape']:.3f} | {m['bias']:+.3f} "
            f"| {extra} |"
        )
    if "model" in test:
        seg = test["model"]
        lines += [
            f"\n## Test, model segment ({seg['products']:,} items with enough history)\n",
            "| method | MAE | RMSE | WAPE | bias | WAPE h=1 | WAPE h=7 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in seg["methods"].items():
            bh = m["by_horizon"]
            lines.append(
                f"| {name} | {m['mae']:.3f} | {m['rmse']:.2f} | {m['wape']:.3f} "
                f"| {m['bias']:+.3f} | {bh.get(1, float('nan')):.3f} "
                f"| {bh.get(7, float('nan')):.3f} |"
            )
    lines += [
        "\n## Test: total demand over the next 7 days (all items)\n",
        "| method | MAE | WAPE | bias |",
        "|---|---:|---:|---:|",
    ]
    for name, m in report["test_cumulative_7day"]["all"].items():
        lines.append(f"| {name} | {m['mae']:.2f} | {m['wape']:.3f} | {m['bias']:+.3f} |")
    lines += [
        "\n## Reading these numbers\n",
        "- RMSSE here is item-level over rolling 1-7 day forecasts for one store. It is not "
        "the competition's WRMSSE (12 aggregation levels, one 28-day forecast), so it is not "
        "comparable with the Kaggle leaderboard.",
        "- M5-specific signals (SNAP days, events) are not used; `invoices` and `customers` "
        "are missing because M5 has no transaction-level data.",
        f"- Runtime: {report['runtime_seconds']:.0f} s.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional M5 benchmark")
    parser.add_argument("--zip", type=Path, default=ZIP_PATH)
    parser.add_argument("--download", action="store_true", help="fetch the zip if missing")
    parser.add_argument("--store", default="CA_1")
    parser.add_argument("--max-items", type=int, help="first N items only (quick runs)")
    parser.add_argument("--train-origin-days", type=int, default=730)
    parser.add_argument("--configs", nargs="+", choices=sorted(T.CONFIGS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.zip.exists():
        if not args.download:
            print(f"{args.zip} not found; rerun with --download", file=sys.stderr)
            return 2
        download(args.zip)
    daily = load_store(args.zip, args.store, args.max_items)
    configs = {k: T.CONFIGS[k] for k in args.configs} if args.configs else T.CONFIGS
    report = run(daily, train_origin_days=args.train_origin_days, configs=configs)
    report["dataset"] = {"store": args.store, **report["dataset"]}
    if args.max_items:
        report["dataset"]["max_items"] = args.max_items
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_MD.write_text(render_markdown(report))
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
