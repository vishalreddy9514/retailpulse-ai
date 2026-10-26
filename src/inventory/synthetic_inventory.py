"""Generate a clearly labelled SYNTHETIC inventory snapshot.

UCI Online Retail II has no stock levels or supplier lead times. This module invents
them, reproducibly, from ``config/inventory_assumptions.toml``, so the stockout-risk
logic has something to run on. None of the values is real retailer inventory. Every row
carries ``is_synthetic = True`` and the snapshot records the assumptions it used.

Per product on sale at the snapshot date (end of a trading day):

* ``lead_time_days``: drawn from the configured distribution (trading days).
* ``mean_daily_demand`` / ``std_daily_demand``: *observed* units/day over the last
  ``demand_window_days`` trading days up to the snapshot date (fewer if newer).
* ``safety_stock`` = ceil(z * std * sqrt(lead time)), with z for the service level.
* ``reorder_point`` = ceil(mean * lead time + safety stock).
* ``current_stock``: uniform(cover_days_min, cover_days_max) days of mean demand, or
  uniform{0..zero_demand_stock_max} when there was no recent demand.

Randomness is seeded per (seed, product, snapshot date). A product's values don't depend
on which other products exist, and the same inputs always give the same snapshot.

Usage::

    python -m src.inventory.synthetic_inventory --as-of 2011-12-08
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
import zlib
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import config
from src.processing.aggregate_sales import PRODUCT_DAILY_PARQUET, PRODUCT_SUMMARY_PARQUET

ASSUMPTIONS_PATH = config.PROJECT_ROOT / "config" / "inventory_assumptions.toml"
INVENTORY_DIR = config.DATA_DIR / "inventory"
SYNTHETIC_LABEL = "SYNTHETIC - generated from config/inventory_assumptions.toml, not real stock"


def load_assumptions(path: Path = ASSUMPTIONS_PATH) -> dict:
    with path.open("rb") as fh:
        a = tomllib.load(fh)
    lt = a["lead_time"]
    if len(lt["days"]) != len(lt["weights"]) or not math.isclose(sum(lt["weights"]), 1.0):
        raise ValueError("lead_time.days and weights must align and weights must sum to 1")
    if not 0.5 < a["policy"]["service_level"] < 1:
        raise ValueError("policy.service_level must be in (0.5, 1)")
    return a


def assumptions_fingerprint(assumptions: dict) -> str:
    blob = json.dumps(assumptions, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _rng(seed: int, stock_code: str, as_of: pd.Timestamp) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(stock_code.encode()), as_of.toordinal()])


def generate(
    stock_codes: np.ndarray,
    mean_daily: np.ndarray,
    std_daily: np.ndarray,
    as_of: pd.Timestamp,
    assumptions: dict,
) -> pd.DataFrame:
    """Synthetic snapshot from per-product observed demand statistics at ``as_of``."""
    gen, lt_cfg, policy = assumptions["generation"], assumptions["lead_time"], assumptions["policy"]
    z = NormalDist().inv_cdf(policy["service_level"])
    lead, stock = [], []
    for code, mean in zip(stock_codes, mean_daily, strict=True):
        rng = _rng(gen["seed"], str(code), as_of)
        lead.append(int(rng.choice(lt_cfg["days"], p=lt_cfg["weights"])))
        if mean > 0:
            cover = rng.uniform(gen["cover_days_min"], gen["cover_days_max"])
            stock.append(int(round(cover * mean)))
        else:
            stock.append(int(rng.integers(0, gen["zero_demand_stock_max"] + 1)))
    lead = np.array(lead)
    std = np.nan_to_num(std_daily, nan=0.0)
    mean = np.nan_to_num(mean_daily, nan=0.0)
    safety = np.ceil(z * std * np.sqrt(lead)).astype(int)
    reorder = np.ceil(mean * lead + safety).astype(int)
    return pd.DataFrame(
        {
            "stock_code": stock_codes,
            "as_of": as_of,
            "current_stock": np.array(stock, dtype=int),
            "lead_time_days": lead,
            "mean_daily_demand": np.round(mean, 3),
            "std_daily_demand": np.round(std, 3),
            "safety_stock": safety,
            "reorder_point": reorder,
            "service_level": policy["service_level"],
            "is_synthetic": True,
            "data_label": SYNTHETIC_LABEL,
            "assumptions_version": assumptions_fingerprint(assumptions),
        }
    )


def demand_stats(product_daily: pd.DataFrame, as_of: pd.Timestamp, window: int) -> pd.DataFrame:
    """Observed mean/std of daily units over the last ``window`` trading days up to ``as_of``."""
    full = product_daily.loc[~product_daily["is_partial_day"] & (product_daily["date"] <= as_of)]
    days = np.sort(full["date"].unique())
    if len(days) == 0 or days[-1] != np.datetime64(as_of):
        raise ValueError(f"{as_of.date()} is not a full trading day in the data")
    recent = full[full["date"] >= days[max(len(days) - window, 0)]]
    g = recent.groupby("stock_code")["units_sold"]
    return pd.DataFrame({"mean_daily": g.mean(), "std_daily": g.std(ddof=0)}).reset_index()


def snapshot(
    product_daily: pd.DataFrame, as_of: str | pd.Timestamp, assumptions: dict | None = None
) -> pd.DataFrame:
    assumptions = assumptions or load_assumptions()
    as_of = pd.Timestamp(as_of)
    stats = demand_stats(product_daily, as_of, assumptions["policy"]["demand_window_days"])
    return generate(
        stats["stock_code"].to_numpy(),
        stats["mean_daily"].to_numpy(),
        stats["std_daily"].to_numpy(),
        as_of,
        assumptions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a SYNTHETIC inventory snapshot")
    parser.add_argument("--as-of", default=None, help="snapshot date (default: last full day)")
    args = parser.parse_args(argv)
    daily = pd.read_parquet(PRODUCT_DAILY_PARQUET)
    as_of = args.as_of or daily.loc[~daily["is_partial_day"], "date"].max()
    snap = snapshot(daily, as_of)
    names = pd.read_parquet(PRODUCT_SUMMARY_PARQUET, columns=["stock_code", "product_description"])
    snap = snap.merge(names, on="stock_code", how="left")
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = INVENTORY_DIR / f"synthetic_inventory_{pd.Timestamp(as_of):%Y-%m-%d}.csv"
    snap.to_csv(out, index=False)
    print(SYNTHETIC_LABEL)
    print(snap.describe().loc[["mean", "50%", "max"]].round(1).to_string())
    print(f"wrote {len(snap):,} products to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
