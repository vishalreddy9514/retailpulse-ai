"""Synthetic inventory, stockout risk and approval workflow tests (synthetic inputs only)."""

from __future__ import annotations

import copy
import math
import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.inventory.approvals import ApprovalError, ApprovalStore
from src.inventory.risk_engine import ErrorModel, _survival, assess
from src.inventory.synthetic_inventory import (
    SYNTHETIC_LABEL,
    demand_stats,
    generate,
    load_assumptions,
    snapshot,
)
from src.processing.trading_calendar import expected_trading_days

ASSUMPTIONS = load_assumptions()
ORIGIN = pd.Timestamp("2011-12-08")


def daily_frame(days=80) -> pd.DataFrame:
    dates = [pd.Timestamp("2011-08-01")] + expected_trading_days("2011-08-01", days - 1)
    rows = []
    for i, d in enumerate(dates):
        rows.append(("STEADY", d, 10, False))
        rows.append(("SLOW", d, 1 if i % 5 == 0 else 0, False))
        rows.append(("DEAD", d, 0, False))
    df = pd.DataFrame(rows, columns=["stock_code", "date", "units_sold", "is_partial_day"])
    return df


# -- synthetic inventory -------------------------------------------------------------------


def test_snapshot_is_labelled_deterministic_and_within_assumptions():
    daily = daily_frame()
    as_of = daily["date"].max()
    a = snapshot(daily, as_of, ASSUMPTIONS)
    b = snapshot(daily, as_of, ASSUMPTIONS)
    pd.testing.assert_frame_equal(a, b)
    assert a["is_synthetic"].all() and (a["data_label"] == SYNTHETIC_LABEL).all()
    assert set(a["lead_time_days"]) <= set(ASSUMPTIONS["lead_time"]["days"])
    steady = a.set_index("stock_code").loc["STEADY"]
    gen = ASSUMPTIONS["generation"]
    assert gen["cover_days_min"] * 10 <= steady["current_stock"] <= gen["cover_days_max"] * 10
    dead = a.set_index("stock_code").loc["DEAD"]
    assert 0 <= dead["current_stock"] <= gen["zero_demand_stock_max"]


def test_safety_stock_and_reorder_point_formulas():
    z = NormalDist().inv_cdf(ASSUMPTIONS["policy"]["service_level"])
    snap = generate(np.array(["X"]), np.array([4.0]), np.array([2.0]), ORIGIN, ASSUMPTIONS)
    lt = snap["lead_time_days"].item()
    assert snap["safety_stock"].item() == math.ceil(z * 2.0 * math.sqrt(lt))
    assert snap["reorder_point"].item() == math.ceil(4.0 * lt + snap["safety_stock"].item())


def test_product_values_do_not_depend_on_other_products():
    one = generate(np.array(["A"]), np.array([5.0]), np.array([1.0]), ORIGIN, ASSUMPTIONS)
    many = generate(
        np.array(["Z", "A", "B"]),
        np.array([1.0, 5.0, 2.0]),
        np.array([1.0, 1.0, 1.0]),
        ORIGIN,
        ASSUMPTIONS,
    )
    cols = ["current_stock", "lead_time_days", "safety_stock", "reorder_point"]
    assert one[cols].iloc[0].tolist() == many[cols].iloc[1].tolist()


def test_demand_stats_use_only_history_up_to_snapshot():
    daily = daily_frame()
    as_of = sorted(daily["date"].unique())[60]
    changed = daily.copy()
    changed.loc[changed["date"] > as_of, "units_sold"] = 999
    pd.testing.assert_frame_equal(demand_stats(daily, as_of, 56), demand_stats(changed, as_of, 56))
    with pytest.raises(ValueError):
        demand_stats(daily, pd.Timestamp("2011-08-06"), 56)  # a Saturday


def test_invalid_assumptions_are_rejected(tmp_path):
    bad = tmp_path / "a.toml"
    bad.write_text(
        "[generation]\nseed=1\ncover_days_min=0\ncover_days_max=1\nzero_demand_stock_max=1\n"
        "[lead_time]\ndays=[3,5]\nweights=[0.5,0.6]\n"
        "[policy]\nservice_level=0.95\ndemand_window_days=56\nreview_period_days=7\n"
        "[risk]\nhigh=0.5\nmedium=0.2\n"
    )
    with pytest.raises(ValueError, match="weights"):
        load_assumptions(bad)


# -- error model ---------------------------------------------------------------------------


def cum_frame(forecast: float, actuals: list[float]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(actuals):
        for h in range(1, 8):
            rows.append((f"P{i}", pd.Timestamp("2011-10-03"), h, a * h / 7, forecast * h / 7))
    return pd.DataFrame(
        rows, columns=["stock_code", "origin_date", "h", "cum_actual", "cum_forecast"]
    )


def test_survival_is_exact_with_point_mass_at_zero():
    sample = np.sort(np.array([0, 0, 0, 2, 5.0]))
    assert _survival(sample, np.array([0.0]))[0] == pytest.approx(0.4)
    assert _survival(sample, np.array([2.0]))[0] == pytest.approx(0.2)
    assert _survival(sample, np.array([5.0]))[0] == 0.0


def test_error_model_reproduces_share_of_nonzero_demand():
    actuals = [0.0] * 70 + [7.0] * 30  # 30% of products had any demand
    em = ErrorModel.fit(cum_frame(1.0, actuals))
    assert em.prob_exceeds(7, np.array([1.0]), np.array([0.0]))[0] == pytest.approx(0.3)
    probs = em.prob_exceeds(7, np.array([1.0, 1.0, 1.0]), np.array([0.0, 3.0, 10.0]))
    assert probs[0] >= probs[1] >= probs[2]  # more stock, lower risk


def test_error_model_zero_forecast_uses_actual_demand():
    em = ErrorModel.fit(cum_frame(0.0, [0.0] * 90 + [4.0] * 10))
    assert em.prob_exceeds(7, np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.1)
    assert em.prob_exceeds(7, np.array([0.0]), np.array([4.0]))[0] == 0.0


# -- risk assessment -----------------------------------------------------------------------


def forecasts_for(rates: dict[str, float]) -> pd.DataFrame:
    targets = expected_trading_days(ORIGIN, 7)
    rows = [
        (code, ORIGIN, t, h, rate, "lightgbm", "test-model")
        for code, rate in rates.items()
        for h, t in enumerate(targets, 1)
    ]
    cols = ["stock_code", "forecast_origin", "target_date", "horizon_trading_days",
            "forecast_units", "method", "model_version"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def inventory_for(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["stock_code", "current_stock", "lead_time_days", "safety_stock", "reorder_point"],
    )


@pytest.fixture
def errors():
    return ErrorModel.fit(cum_frame(70.0, [70.0 * r for r in np.linspace(0.5, 1.5, 100)]))


def test_projection_stockout_date_and_reorder(errors):
    fc = forecasts_for({"LOW": 10.0, "HIGH": 10.0})
    inv = inventory_for([("LOW", 25, 5, 8, 58), ("HIGH", 500, 5, 8, 58)])
    risk = assess(fc, inv, errors, ASSUMPTIONS).set_index("stock_code")

    low = risk.loc["LOW"]
    assert low["forecast_units_7d"] == 70.0
    assert low["projected_stock_7d"] == -45.0
    assert low["expected_stockout_date"] == expected_trading_days(ORIGIN, 3)[-1]  # 30 > 25 on day 3
    assert low["reorder_suggested"] and low["suggestion_status"] == "pending_approval"
    # Order up to: 10/day x (5 lead + 7 review) + 8 safety - 25 on hand = 103.
    assert low["suggested_order_qty"] == 103
    assert low["stockout_before_delivery_likely"]

    high = risk.loc["HIGH"]
    assert not high["reorder_suggested"] and high["suggested_order_qty"] == 0
    assert high["suggestion_status"] == "none" and high["suggestion_id"] == ""
    assert high["stockout_probability_7d"] < low["stockout_probability_7d"]


def test_probabilities_are_monotone_in_horizon_and_labelled(errors):
    risk = assess(
        forecasts_for({"A": 3.0}), inventory_for([("A", 10, 3, 2, 11)]), errors, ASSUMPTIONS
    )
    row = risk.iloc[0]
    assert row["stockout_probability_next_day"] <= row["stockout_probability_7d"]
    assert row["risk_level"] in {"low", "medium", "high"}
    assert "synthetic" in row["value_types"] and row["inventory_data_label"] == SYNTHETIC_LABEL
    assert row["model_version"] == "test-model"


def test_suggestion_ids_are_deterministic_and_depend_on_assumptions(errors):
    fc, inv = forecasts_for({"LOW": 10.0}), inventory_for([("LOW", 1, 5, 8, 58)])
    a = assess(fc, inv, errors, ASSUMPTIONS)["suggestion_id"].item()
    assert a == assess(fc, inv, errors, ASSUMPTIONS)["suggestion_id"].item()
    other = copy.deepcopy(ASSUMPTIONS)
    other["policy"]["service_level"] = 0.9
    assert a != assess(fc, inv, errors, other)["suggestion_id"].item()


def test_missing_inventory_row_is_an_error(errors):
    with pytest.raises(ValueError, match="no synthetic inventory"):
        assess(
            forecasts_for({"A": 1.0, "B": 1.0}),
            inventory_for([("A", 1, 3, 1, 4)]),
            errors,
            ASSUMPTIONS,
        )


# -- approvals -----------------------------------------------------------------------------


@pytest.fixture
def store_and_id(errors, tmp_path):
    risk = assess(
        forecasts_for({"LOW": 10.0}), inventory_for([("LOW", 1, 5, 8, 58)]), errors, ASSUMPTIONS
    )
    store = ApprovalStore(tmp_path / "approvals.sqlite")
    assert store.register(risk) == 1
    assert store.register(risk) == 0  # idempotent
    yield store, risk["suggestion_id"].item(), tmp_path / "approvals.sqlite"
    store.close()


def test_suggestion_starts_pending_and_needs_a_named_person(store_and_id):
    store, sid, _ = store_and_id
    assert store.get(sid)["status"] == "pending_approval"
    with pytest.raises(ApprovalError, match="named person"):
        store.decide(sid, "approved", "   ")
    with pytest.raises(ApprovalError, match="decision"):
        store.decide(sid, "auto", "A. Buyer")


def test_decision_is_final_and_audited(store_and_id):
    store, sid, _ = store_and_id
    out = store.decide(sid, "approved", "A. Buyer", "checked against supplier")
    assert out["status"] == "approved"
    assert [h["action"] for h in out["history"]] == ["registered", "approved"]
    assert out["history"][-1]["actor"] == "A. Buyer"
    assert out["details"]["inventory_data_label"] == SYNTHETIC_LABEL
    with pytest.raises(ApprovalError, match="already approved"):
        store.decide(sid, "rejected", "B. Buyer")
    with pytest.raises(ApprovalError, match="unknown"):
        store.decide("nope", "approved", "A. Buyer")


def test_audit_log_is_append_only(store_and_id):
    _, _, path = store_and_id
    conn = sqlite3.connect(path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM audit_log")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE audit_log SET actor='someone else'")
    conn.close()
