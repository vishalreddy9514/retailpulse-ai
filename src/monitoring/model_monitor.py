"""Monitor the deployed forecasts: live accuracy, input drift and data freshness.

Run it against a forecast made at an earlier origin, once the actuals for its target days
have arrived (from the batch tables or the stream)::

    python -m src.monitoring.model_monitor --origin 2011-11-15

Checks (each ``ok`` / ``degraded`` / ``insufficient_data``):

* ``forecast_accuracy``: WAPE of the forecast against the actual units on each target
  day, for single days and for 7-day totals. "Degraded" when it is more than
  ``TOLERANCE`` (relative) worse than the held-out test WAPE. Products launched after the
  origin are excluded, because the model never forecasts them. Partial days are excluded.
* ``input_drift``: population stability index (PSI) of store-wide daily units and of the
  share of zero-demand product-days, last 28 trading days vs the 112 before. PSI > 0.25 is
  flagged. In retail this also fires on genuine seasonality (the Q4 peak), so treat it as
  "look at this", not as an error.
* ``freshness``: the data must reach the forecast's last target day.

Accuracy is only meaningful out of sample. If the model that produced the forecast was
trained on any of its target days (``training_window.last_target`` in the model metadata),
the check reports ``in_sample`` instead of ``ok``. ``--from-backtest`` scores the
held-out forecasts of the phase 4 test model, which was trained only on data before them.

Writes ``reports/monitoring_report.{json,md}``. The API exposes the results as Prometheus
gauges that the alert rules in ``monitoring/alerts.yml`` watch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.forecasting.metrics import all_metrics, wape
from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET

TOLERANCE = 0.15
PSI_ALERT = 0.25
RECENT_DAYS, REFERENCE_DAYS = 28, 112
REPORT_JSON = config.REPORTS_DIR / "monitoring_report.json"
REPORT_MD = config.REPORTS_DIR / "monitoring_report.md"
FORECAST_EVAL_JSON = config.REPORTS_DIR / "forecast_evaluation.json"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index with quantile bins from the reference sample."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def forecast_performance(forecasts: pd.DataFrame, product_daily: pd.DataFrame) -> dict:
    origin = pd.Timestamp(forecasts["forecast_origin"].iloc[0])
    full = product_daily.loc[~product_daily["is_partial_day"]]
    actual = full[["stock_code", "date", "units_sold"]].rename(columns={"date": "target_date"})
    joined = forecasts.merge(actual, on=["stock_code", "target_date"], how="inner")
    joined = joined[
        joined["method"] == "lightgbm"
    ]  # model-forecast products, as in the test metric
    targets = sorted(forecasts["target_date"].unique())
    covered = sorted(joined["target_date"].unique())
    if joined.empty:
        return {"origin": str(origin.date()), "target_days": len(targets), "target_days_with_actuals": 0,
                "wape": {}, "metrics": {}}  # fmt: skip
    daily = all_metrics(joined["units_sold"].to_numpy(), joined["forecast_units"].to_numpy())
    week = joined.groupby("stock_code")[["units_sold", "forecast_units"]].sum()
    complete = len(covered) == len(targets)
    out = {
        "origin": str(origin.date()),
        "target_days": len(targets),
        "target_days_with_actuals": len(covered),
        "products": int(joined["stock_code"].nunique()),
        "metrics": {"daily": daily},
        "wape": {"daily": daily["wape"]},
    }
    if complete:
        out["metrics"]["7d"] = all_metrics(
            week["units_sold"].to_numpy(), week["forecast_units"].to_numpy()
        )
        out["wape"]["7d"] = round(wape(week["units_sold"], week["forecast_units"]), 4)
    return out


def input_drift(product_daily: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    full = product_daily.loc[~product_daily["is_partial_day"] & (product_daily["date"] <= as_of)]
    days = np.sort(full["date"].unique())
    recent_days = days[-RECENT_DAYS:]
    ref_days = days[-(RECENT_DAYS + REFERENCE_DAYS) : -RECENT_DAYS]
    totals = full.groupby("date")["units_sold"].sum()
    zero_share = full.assign(z=full["units_sold"].eq(0)).groupby("date")["z"].mean()
    values = {
        "store_daily_units": psi(
            totals.loc[ref_days].to_numpy(), totals.loc[recent_days].to_numpy()
        ),
        "zero_demand_share": psi(
            zero_share.loc[ref_days].to_numpy(), zero_share.loc[recent_days].to_numpy()
        ),
    }
    return {
        "recent_window": [
            str(pd.Timestamp(recent_days[0]).date()),
            str(pd.Timestamp(recent_days[-1]).date()),
        ],
        "reference_window": [
            str(pd.Timestamp(ref_days[0]).date()),
            str(pd.Timestamp(ref_days[-1]).date()),
        ],
        "psi": {k: round(v, 4) for k, v in values.items()},
        "means": {
            "store_daily_units": {
                "reference": round(float(totals.loc[ref_days].mean()), 1),
                "recent": round(float(totals.loc[recent_days].mean()), 1),
            },
            "zero_demand_share": {
                "reference": round(float(zero_share.loc[ref_days].mean()), 4),
                "recent": round(float(zero_share.loc[recent_days].mean()), 4),
            },
        },  # fmt: skip
    }


def backtest_forecasts(backtest: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Held-out forecasts made at ``origin`` by the test model (trained before the test window)."""
    bt = backtest[backtest["origin_date"] == pd.Timestamp(origin)]
    if bt.empty:
        raise ValueError(f"no backtest forecasts at origin {origin}")
    bt = bt[bt["segment"] != "launched_after_origin"]
    return pd.DataFrame(
        {
            "stock_code": bt["stock_code"],
            "forecast_origin": bt["origin_date"],
            "target_date": bt["target_date"],
            "horizon_trading_days": bt["horizon"],
            "forecast_units": bt["forecast_pipeline"],
            "method": np.where(bt["segment"] == "model", "lightgbm", "fallback"),
        }
    )


def run(
    forecasts: pd.DataFrame,
    product_daily: pd.DataFrame,
    baseline: dict,
    trained_through: str | None = None,
) -> dict:
    origin = pd.Timestamp(forecasts["forecast_origin"].iloc[0])
    perf = forecast_performance(forecasts, product_daily)
    drift = input_drift(product_daily, origin)
    last_target = pd.Timestamp(forecasts["target_date"].max())
    last_full_day = product_daily.loc[~product_daily["is_partial_day"], "date"].max()

    base = {
        "daily": baseline["test"]["segments"]["model"]["methods"]["lightgbm"]["wape"],
        "7d": baseline["test_cumulative_7day"]["model"]["lightgbm"]["wape"],
    }
    in_sample = (
        trained_through is not None
        and pd.Timestamp(trained_through) >= forecasts["target_date"].min()
    )
    if perf["wape"] and in_sample:
        accuracy = {"status": "in_sample", "baseline_wape": base, "trained_through": trained_through,
                    "note": "The model was trained on these target days; this WAPE is not an accuracy measurement."}  # fmt: skip
    elif perf["wape"]:
        worse = {k: v > base[k] * (1 + TOLERANCE) for k, v in perf["wape"].items()}
        accuracy = {"status": "degraded" if any(worse.values()) else "ok", "baseline_wape": base,
                    "tolerance": TOLERANCE, "worse_than_baseline": worse}  # fmt: skip
    else:
        accuracy = {"status": "insufficient_data", "baseline_wape": base}
    drifted = {k: v > PSI_ALERT for k, v in drift["psi"].items()}
    return {
        "forecast_origin": str(origin.date()),
        "data_note": "Historical UCI Online Retail II replay; inventory is synthetic.",
        "forecast_performance": perf,
        "input_drift": drift,
        "checks": {
            "forecast_accuracy": accuracy,
            "input_drift": {
                "status": "degraded" if any(drifted.values()) else "ok",
                "threshold": PSI_ALERT,
                "drifted": drifted,
            },  # fmt: skip
            "freshness": {
                "status": "ok" if last_full_day >= last_target else "insufficient_data",
                "last_full_day": str(last_full_day.date()),
                "last_target_day": str(last_target.date()),
            },
        },
    }


def render(report: dict) -> str:
    perf, drift, checks = report["forecast_performance"], report["input_drift"], report["checks"]
    lines = [
        "# Monitoring report\n",
        f"Forecast origin {report['forecast_origin']} ({report.get('forecast_source', '')}); {report['data_note']}\n",
        "| check | status |",
        "|---|---|",
        *[f"| {k} | {v['status']} |" for k, v in checks.items()],
        "",
        f"## Forecast accuracy against later actuals ({perf.get('products', 0):,} model-forecast products, "
        f"{perf['target_days_with_actuals']}/{perf['target_days']} target days with actuals)\n",
        "| window | WAPE now | held-out test WAPE | MAE | bias |",
        "|---|---:|---:|---:|---:|",
    ]
    base = checks["forecast_accuracy"].get("baseline_wape", {})
    for window, m in perf.get("metrics", {}).items():
        lines.append(
            f"| {window} | {m['wape']:.3f} | {base.get(window, float('nan')):.3f} | {m['mae']:.2f} | {m['bias']:+.3f} |"
        )
    lines += [
        "",
        f"## Input drift (recent {drift['recent_window'][0]} to {drift['recent_window'][1]} vs "
        f"reference {drift['reference_window'][0]} to {drift['reference_window'][1]})\n",
        "| feature | PSI | reference mean | recent mean |",
        "|---|---:|---:|---:|",
        *[
            f"| {k} | {v:.3f} | {drift['means'][k]['reference']} | {drift['means'][k]['recent']} |"
            for k, v in drift["psi"].items()
        ],  # fmt: skip
        "",
        f"PSI above {PSI_ALERT} is flagged. Seasonal peaks also raise PSI, so a flag means 'review', not 'broken'.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor forecast accuracy, drift and freshness")
    parser.add_argument(
        "--origin", required=True, help="origin of a saved forecast (forecasts_<origin>.parquet)"
    )
    parser.add_argument(
        "--from-backtest",
        action="store_true",
        help="score the held-out test-model forecasts at this origin (out of sample)",
    )
    parser.add_argument("--out", type=Path, default=REPORT_JSON)
    args = parser.parse_args(argv)
    if args.from_backtest:
        from src.forecasting.train import BACKTEST_PARQUET, VAL_END

        forecasts = backtest_forecasts(pd.read_parquet(BACKTEST_PARQUET), args.origin)
        trained_through = str(VAL_END.date())  # the test model saw targets up to validation end
    else:
        from src.forecasting.train import META_PATH

        forecasts = pd.read_parquet(config.PROCESSED_DIR / f"forecasts_{args.origin}.parquet")
        trained_through = json.loads(META_PATH.read_text())["training_window"]["last_target"]
    report = run(
        forecasts,
        pd.read_parquet(PRODUCT_DAILY_PARQUET),
        json.loads(FORECAST_EVAL_JSON.read_text()),
        trained_through,
    )
    report["forecast_source"] = (
        "held-out backtest (test model)" if args.from_backtest else "production model"
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(render(report))
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
