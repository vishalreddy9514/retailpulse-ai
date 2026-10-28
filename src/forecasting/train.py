"""Train, evaluate and persist the product-level demand forecasting model.

Protocol (all boundaries are *target* dates, so no training row ever sees an evaluation
target):

* train:       targets up to TRAIN_END
* validation:  TRAIN_END < target <= VAL_END. Used for early stopping and config choice.
* test:        VAL_END < target <= last full trading day. Scored once by a model refit
  on train + validation.

Evaluation is rolling-origin. Every trading day in a window is a forecast origin, and
every horizon from 1 to 7 trading days whose target falls inside the window is scored.
Products are split into segments at each origin:

* ``model``: enough history at the origin (MIN_SELLING_DAYS / MIN_WINDOW_DAYS); LightGBM
* ``fallback``: on sale but too little history; 28-trading-day mean excluding the
  launch day (see ``features.fallback_forecast``)
* ``launched_after_origin``: no history at all at the origin; forecast 0 and reported
  separately, because nothing at the origin could predict a launch.

The production model is refit on everything up to the last full day with the selected
config and iteration count, then saved to ``models/``.

Usage::

    python -m src.forecasting.train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.forecasting import features as F
from src.forecasting.metrics import all_metrics, wape
from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET

logger = logging.getLogger(__name__)

TRAIN_END = pd.Timestamp("2011-06-30")
VAL_END = pd.Timestamp("2011-09-30")
SEED = 20111209

MODELS_DIR = config.PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "demand_lgbm.txt"
META_PATH = MODELS_DIR / "demand_lgbm.json"
BACKTEST_PARQUET = config.PROCESSED_DIR / "forecast_backtest.parquet"
REPORT_JSON = config.REPORTS_DIR / "forecast_evaluation.json"
REPORT_MD = config.REPORTS_DIR / "forecast_evaluation.md"

BASE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "l1",
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
# Small, explicit search space, chosen on validation WAPE. Tweedie and Poisson suit
# non-negative, zero-heavy counts.
CONFIGS = {
    "tweedie_1.2": {"objective": "tweedie", "tweedie_variance_power": 1.2},
    "tweedie_1.5": {"objective": "tweedie", "tweedie_variance_power": 1.5},
    "poisson": {"objective": "poisson"},
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


@dataclass
class Rows:
    t_idx: np.ndarray
    p_idx: np.ndarray
    horizon: np.ndarray
    segment: np.ndarray  # "model" / "fallback" / "launched_after_origin"

    def __len__(self) -> int:
        return len(self.t_idx)

    def take(self, mask: np.ndarray) -> Rows:
        return Rows(self.t_idx[mask], self.p_idx[mask], self.horizon[mask], self.segment[mask])


def training_rows(panel: F.Panel, last_target: pd.Timestamp, seed: int = SEED) -> Rows:
    """One row per model-eligible (origin, product), with a seeded random horizon.

    Using all 7 horizons per pair would mean ~12.5M near-duplicate rows. Sampling one
    keeps every origin and product while covering all horizons evenly.
    """
    t_idx, p_idx = np.nonzero(panel.eligible)
    rng = np.random.default_rng(seed)
    horizon = rng.integers(1, len(F.HORIZONS) + 1, size=len(t_idx))
    target = t_idx + horizon
    last = panel.dates.searchsorted(last_target, side="right") - 1
    keep = target <= last
    seg = np.full(keep.sum(), "model", dtype=object)
    return Rows(t_idx[keep], p_idx[keep], horizon[keep], seg)


def evaluation_rows(panel: F.Panel, window_start: pd.Timestamp, window_end: pd.Timestamp) -> Rows:
    """All (origin, product, horizon) whose target lies in [window_start, window_end]."""
    first = panel.dates.searchsorted(window_start, side="left")
    last = panel.dates.searchsorted(window_end, side="right") - 1
    exists = panel.units.notna().to_numpy()
    parts = []
    for h in F.HORIZONS:
        for target in range(first, last + 1):
            t = target - h
            if t < 0:
                continue
            launched_by_target = exists[target]
            existing = np.nonzero(launched_by_target & exists[t])[0]
            new = np.nonzero(launched_by_target & ~exists[t])[0]
            eligible = panel.eligible[t, existing]
            segs = np.where(eligible, "model", "fallback").astype(object)
            p = np.concatenate([existing, new])
            seg = np.concatenate([segs, np.full(len(new), "launched_after_origin", dtype=object)])
            parts.append((np.full(len(p), t), p, np.full(len(p), h), seg))
    cat = [np.concatenate(x) for x in zip(*parts, strict=True)]
    return Rows(*cat)


def target_dates(panel: F.Panel, rows: Rows) -> pd.DatetimeIndex:
    return panel.dates[rows.t_idx + rows.horizon]


def targets(panel: F.Panel, rows: Rows) -> np.ndarray:
    return panel.units.to_numpy()[rows.t_idx + rows.horizon, rows.p_idx]


def design(panel: F.Panel, rows: Rows) -> pd.DataFrame:
    return F.feature_frame(panel, rows.t_idx, rows.p_idx, rows.horizon, target_dates(panel, rows))


def fit(params: dict, train: lgb.Dataset, rounds: int, valid: lgb.Dataset | None = None):
    callbacks = []
    kwargs = {}
    if valid is not None:
        callbacks.append(lgb.early_stopping(EARLY_STOPPING, verbose=False))
        kwargs = {"valid_sets": [valid], "valid_names": ["validation"]}
    return lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)


def select_config(
    dtrain: lgb.Dataset,
    dval: lgb.Dataset,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    configs: dict[str, dict] | None = None,
    max_rounds: int | None = None,
) -> tuple[dict, str]:
    """Fit each config with early stopping on validation; pick the lowest validation WAPE."""
    configs = CONFIGS if configs is None else configs
    max_rounds = MAX_ROUNDS if max_rounds is None else max_rounds
    selection = {}
    for name, cfg in configs.items():
        booster = fit({**BASE_PARAMS, **cfg}, dtrain, max_rounds, dval)
        score = wape(y_val, booster.predict(x_val, num_iteration=booster.best_iteration))
        selection[name] = {
            "best_iteration": int(booster.best_iteration),
            "validation_wape": round(score, 4),
        }
        logger.info(
            "Config %s: best_iteration=%d validation WAPE=%.4f", name, booster.best_iteration, score
        )
    return selection, min(selection, key=lambda k: selection[k]["validation_wape"])


def predict_rows(panel: F.Panel, rows: Rows, booster: lgb.Booster | None) -> dict[str, np.ndarray]:
    """Forecasts from every method on the same rows. The 'pipeline' combines the segments."""
    td = target_dates(panel, rows)
    out = {
        "zero": np.zeros(len(rows)),
        "seasonal_naive": F.seasonal_naive(panel, rows.t_idx, rows.p_idx, td),
        "moving_average_28": F.moving_average(panel, rows.t_idx, rows.p_idx, 28),
    }
    model_mask = rows.segment == "model"
    pipeline = F.fallback_forecast(panel, rows.t_idx, rows.p_idx)
    if booster is not None:
        lgbm = np.full(len(rows), np.nan)
        lgbm[model_mask] = booster.predict(design(panel, rows.take(model_mask)))
        out["lightgbm"] = lgbm
        pipeline[model_mask] = lgbm[model_mask]
    pipeline[rows.segment == "launched_after_origin"] = 0.0
    out["pipeline"] = pipeline
    return out


def fingerprint(product_daily: pd.DataFrame) -> str:
    key = product_daily[["stock_code", "date", "units_sold"]].sort_values(["stock_code", "date"])
    hashed = pd.util.hash_pandas_object(key, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def evaluate(panel: F.Panel, rows: Rows, preds: dict[str, np.ndarray]) -> dict:
    """Metrics per segment, method and horizon."""
    y = targets(panel, rows)
    result: dict = {"segments": {}}
    for seg in ("model", "fallback", "launched_after_origin", "all"):
        mask = np.ones(len(rows), bool) if seg == "all" else rows.segment == seg
        if not mask.any():
            continue
        methods = {}
        for name, yhat in preds.items():
            if name == "lightgbm" and seg != "model":
                continue
            methods[name] = all_metrics(y[mask], yhat[mask])
            methods[name]["by_horizon"] = {
                int(h): round(
                    wape(y[mask & (rows.horizon == h)], yhat[mask & (rows.horizon == h)]), 4
                )
                for h in F.HORIZONS
                if (mask & (rows.horizon == h)).any()
            }
        result["segments"][seg] = {
            "rows": int(mask.sum()),
            "products": int(len(np.unique(rows.p_idx[mask]))),
            "actual_units": int(y[mask].sum()),
            "methods": methods,
        }
    return result


def cumulative_7day(backtest: pd.DataFrame) -> dict:
    """Metrics on total demand over the next 7 trading days, per (product, origin).

    Replenishment depends on demand summed over a lead time, not on single days. Only
    origins with all 7 horizons inside the evaluation window are used.
    """
    cols = [c for c in backtest.columns if c.startswith("forecast_")]
    g = backtest.groupby(["stock_code", "origin_date"], observed=True)
    sums = g[["actual_units", *cols]].sum()
    complete = g["horizon"].size() == len(F.HORIZONS)
    segment = g["segment"].first()
    sums, segment = sums[complete], segment[complete]
    result = {}
    for seg in ("model", "fallback", "all"):
        mask = np.ones(len(sums), bool) if seg == "all" else (segment == seg).to_numpy()
        if not mask.any():
            continue
        y = sums["actual_units"].to_numpy()[mask]
        result[seg] = {
            name.removeprefix("forecast_"): all_metrics(y, sums[name].to_numpy()[mask])
            for name in cols
            if not (name == "forecast_lightgbm" and seg != "model")
        }
    return result


def run(product_daily: pd.DataFrame) -> dict:
    panel = F.build_panel(product_daily)
    last_full = panel.dates[-1]
    logger.info(
        "Panel: %d trading days x %d products, last full day %s",
        len(panel.dates),
        len(panel.codes),
        last_full.date(),
    )

    train_rows = training_rows(panel, TRAIN_END)
    val_rows = evaluation_rows(panel, TRAIN_END + pd.Timedelta(days=1), VAL_END)
    val_model = val_rows.take(val_rows.segment == "model")
    logger.info("Training rows: %d, validation model rows: %d", len(train_rows), len(val_model))

    x_train, y_train = design(panel, train_rows), targets(panel, train_rows)
    x_val, y_val = design(panel, val_model), targets(panel, val_model)
    dtrain = lgb.Dataset(x_train, y_train, free_raw_data=False)
    dval = lgb.Dataset(x_val, y_val, reference=dtrain, free_raw_data=False)

    selection, best_name = select_config(dtrain, dval, x_val, y_val)
    params = {**BASE_PARAMS, **CONFIGS[best_name]}
    rounds = selection[best_name]["best_iteration"]

    # Validation metrics for every method (model trained on train only).
    val_booster = fit(params, dtrain, rounds)
    validation = evaluate(panel, val_rows, predict_rows(panel, val_rows, val_booster))

    # Test: refit on train + validation, score once.
    trval_rows = training_rows(panel, VAL_END)
    test_booster = fit(
        params, lgb.Dataset(design(panel, trval_rows), targets(panel, trval_rows)), rounds
    )
    test_rows = evaluation_rows(panel, VAL_END + pd.Timedelta(days=1), last_full)
    test_preds = predict_rows(panel, test_rows, test_booster)
    test = evaluate(panel, test_rows, test_preds)

    # Production: refit on everything up to the last full day.
    all_rows = training_rows(panel, last_full)
    prod_booster = fit(
        params, lgb.Dataset(design(panel, all_rows), targets(panel, all_rows)), rounds
    )

    importance = pd.Series(
        prod_booster.feature_importance("gain"), index=prod_booster.feature_name()
    ).sort_values(ascending=False)

    backtest = pd.DataFrame(
        {
            "stock_code": panel.codes[test_rows.p_idx],
            "origin_date": panel.dates[test_rows.t_idx],
            "target_date": target_dates(panel, test_rows),
            "horizon": test_rows.horizon,
            "segment": test_rows.segment.astype(str),
            "actual_units": targets(panel, test_rows),
            **{f"forecast_{k}": v for k, v in test_preds.items()},
        }
    )
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    data_fp = fingerprint(product_daily)
    meta = {
        "model_version": f"lgbm-{data_fp}-{best_name}",
        "trained_at": trained_at,
        "config": best_name,
        "params": params,
        "num_boost_round": rounds,
        "features": F.FEATURES,
        "horizons": list(F.HORIZONS),
        "eligibility": {
            "min_selling_days": F.MIN_SELLING_DAYS,
            "min_window_days": F.MIN_WINDOW_DAYS,
        },
        "fallback": "28-day mean excluding launch day (zero for products with no history)",
        "training_window": {
            "first_origin": str(panel.dates[0].date()),
            "last_target": str(last_full.date()),
        },
        "excluded_partial_days": sorted(
            str(d.date())
            for d in product_daily.loc[product_daily["is_partial_day"], "date"].unique()
        ),
        "data_fingerprint": data_fp,
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return {
        "meta": meta,
        "booster": prod_booster,
        "backtest": backtest,
        "report": {
            "splits": {
                "train": {"last_target": str(TRAIN_END.date()), "rows": len(train_rows)},
                "validation": {
                    "targets": [str((TRAIN_END + pd.Timedelta(days=1)).date()), str(VAL_END.date())]
                },
                "test": {
                    "targets": [str((VAL_END + pd.Timedelta(days=1)).date()), str(last_full.date())]
                },
            },
            "config_selection": selection,
            "selected_config": best_name,
            "validation": validation,
            "test": test,
            "test_cumulative_7day": cumulative_7day(backtest),
            "feature_importance_gain_top15": {
                k: round(float(v), 1) for k, v in importance.head(15).items()
            },
            "model_version": meta["model_version"],
        },
    }


def render_markdown(report: dict) -> str:
    def table(seg: dict) -> str:
        lines = [
            "| method | MAE | RMSE | WAPE | bias | WAPE h=1 | WAPE h=7 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in seg["methods"].items():
            bh = m["by_horizon"]
            lines.append(
                f"| {name} | {m['mae']:.3f} | {m['rmse']:.2f} | {m['wape']:.3f} | {m['bias']:+.3f} "
                f"| {bh.get(1, float('nan')):.3f} | {bh.get(7, float('nan')):.3f} |"
            )
        return "\n".join(lines)

    parts = [
        "# Forecast evaluation\n",
        "Generated by `python -m src.forecasting.train`. Units: product units per trading day.",
        "Rolling-origin evaluation, every origin in the window, horizons 1-7 trading days.",
        f"Selected config: **{report['selected_config']}** "
        f"(validation WAPE by config: "
        + ", ".join(
            f"{k} {v['validation_wape']:.4f}" for k, v in report["config_selection"].items()
        )
        + ").\n",
    ]
    for split in ("test", "validation"):
        s = report["splits"][split]
        parts.append(f"## {split.title()} (targets {s['targets'][0]} to {s['targets'][1]})\n")
        for seg_name, seg in report[split]["segments"].items():
            parts.append(
                f"### Segment `{seg_name}`: {seg['rows']:,} forecasts, "
                f"{seg['products']:,} products, {seg['actual_units']:,} actual units\n"
            )
            parts.append(table(seg) + "\n")
    parts.append(
        "## Test: total demand over the next 7 trading days\n\n"
        "Sum of horizons 1-7 per product and origin, compared with actual units over the same "
        "days. This is the quantity replenishment decisions depend on.\n"
    )
    for seg_name, methods in report["test_cumulative_7day"].items():
        rows = next(iter(methods.values()))["rows"]
        parts.append(f"### Segment `{seg_name}`: {rows:,} product-origins\n")
        lines = ["| method | MAE | RMSE | WAPE | bias |", "|---|---:|---:|---:|---:|"]
        lines += [
            f"| {n} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['wape']:.3f} | {m['bias']:+.3f} |"
            for n, m in methods.items()
        ]
        parts.append("\n".join(lines) + "\n")
    parts.append("## Feature importance (gain, production model, top 15)\n")
    parts.append("| feature | gain |\n|---|---:|")
    parts.extend(f"| {k} | {v:,.0f} |" for k, v in report["feature_importance_gain_top15"].items())
    return "\n".join(parts) + "\n"


def save(result: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result["booster"].save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps(result["meta"], indent=2) + "\n")
    BACKTEST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    result["backtest"].to_parquet(BACKTEST_PARQUET, index=False)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result["report"], indent=2) + "\n")
    REPORT_MD.write_text(render_markdown(result["report"]))
    logger.info("Saved model to %s and reports to %s", MODEL_PATH, REPORT_MD)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Train and evaluate the demand model").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(pd.read_parquet(PRODUCT_DAILY_PARQUET))
    save(result)
    test = result["report"]["test"]["segments"]
    for seg in ("model", "all"):
        print(f"test segment {seg}:")
        for name, m in test[seg]["methods"].items():
            print(
                f"  {name:<18} MAE {m['mae']:8.3f}  RMSE {m['rmse']:8.2f}  "
                f"WAPE {m['wape']:.3f}  bias {m['bias']:+.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
