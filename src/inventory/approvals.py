"""Human approval of reorder suggestions, with an audit trail.

RetailPulse never places orders. A reorder suggestion from ``risk_engine`` is
``pending_approval`` until a named person approves or rejects it. Every decision is
written to an append-only audit table, with the suggestion's numbers, the forecast model
version and the synthetic-inventory assumptions it was based on. A decision cannot be
changed silently: a suggestion is decided once, and a new suggestion needs a new forecast
origin or new assumptions (which give a new ``suggestion_id``).

"Approved" means a person has accepted the suggestion for *manual* ordering outside this
system. Quantities rest on synthetic stock levels, so in this project they illustrate
the workflow and are not real purchase decisions.

Usage::

    python -m src.inventory.approvals list
    python -m src.inventory.approvals approve <suggestion_id> --by "A. Buyer" --note "checked"
    python -m src.inventory.approvals reject  <suggestion_id> --by "A. Buyer" --note "overstock"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.inventory.synthetic_inventory import INVENTORY_DIR

DB_PATH = INVENTORY_DIR / "approvals.sqlite"
DECISIONS = ("approved", "rejected")
COMMANDS = {"approve": "approved", "reject": "rejected"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS suggestions (
    suggestion_id TEXT PRIMARY KEY,
    forecast_origin TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    suggested_order_qty INTEGER NOT NULL,
    details TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending_approval',
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id TEXT NOT NULL REFERENCES suggestions(suggestion_id),
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    note TEXT,
    at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_log_append_only_u BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_append_only_d BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""

DETAIL_COLUMNS = [
    "product_description",
    "current_stock",
    "lead_time_days",
    "safety_stock",
    "reorder_point",
    "forecast_units_7d",
    "stockout_probability_7d",
    "expected_stockout_date",
    "earliest_delivery_date",
    "reorder_reason",
    "model_version",
    "assumptions_version",
    "inventory_data_label",
]


class ApprovalError(ValueError):
    pass


class ApprovalStore:
    def __init__(self, path: Path | str = DB_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def register(self, risk: pd.DataFrame, actor: str = "risk_engine") -> int:
        """Register new suggestions as pending. Re-registering the same id does nothing."""
        now = datetime.now(UTC).isoformat()
        added = 0
        rows = risk[risk["reorder_suggested"]]
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for r in rows.itertuples(index=False):
                rec = r._asdict()
                details = {c: _jsonable(rec.get(c)) for c in DETAIL_COLUMNS if c in rec}
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO suggestions"
                    " (suggestion_id, forecast_origin, stock_code, suggested_order_qty, details,"
                    " registered_at) VALUES (?,?,?,?,?,?)",
                    (
                        rec["suggestion_id"],
                        str(pd.Timestamp(rec["forecast_origin"]).date()),
                        rec["stock_code"],
                        int(rec["suggested_order_qty"]),
                        json.dumps(details),
                        now,
                    ),
                )
                if cur.rowcount:
                    added += 1
                    self._log(rec["suggestion_id"], "registered", actor, None, now)
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return added

    def decide(
        self, suggestion_id: str, decision: str, actor: str, note: str | None = None
    ) -> dict:
        if decision not in DECISIONS:
            raise ApprovalError(f"decision must be one of {DECISIONS}")
        if not actor or not actor.strip():
            raise ApprovalError("a named person must make the decision")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT status FROM suggestions WHERE suggestion_id=?", (suggestion_id,)
            ).fetchone()
            if row is None:
                raise ApprovalError(f"unknown suggestion {suggestion_id}")
            if row[0] != "pending_approval":
                raise ApprovalError(f"suggestion {suggestion_id} is already {row[0]}")
            now = datetime.now(UTC).isoformat()
            self.conn.execute(
                "UPDATE suggestions SET status=? WHERE suggestion_id=?", (decision, suggestion_id)
            )
            self._log(suggestion_id, decision, actor.strip(), note, now)
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return self.get(suggestion_id)

    def get(self, suggestion_id: str) -> dict:
        row = self.conn.execute(
            "SELECT suggestion_id, forecast_origin, stock_code, suggested_order_qty, details,"
            " status, registered_at FROM suggestions WHERE suggestion_id=?",
            (suggestion_id,),
        ).fetchone()
        if row is None:
            raise ApprovalError(f"unknown suggestion {suggestion_id}")
        keys = ["suggestion_id", "forecast_origin", "stock_code", "suggested_order_qty",
                "details", "status", "registered_at"]  # fmt: skip
        out = dict(zip(keys, row, strict=True))
        out["details"] = json.loads(out["details"])
        out["history"] = [
            dict(zip(["action", "actor", "note", "at"], h, strict=True))
            for h in self.conn.execute(
                "SELECT action, actor, note, at FROM audit_log WHERE suggestion_id=? ORDER BY id",
                (suggestion_id,),
            )
        ]
        return out

    def list(self, status: str | None = "pending_approval") -> pd.DataFrame:
        q = (
            "SELECT suggestion_id, forecast_origin, stock_code, suggested_order_qty, status"
            " FROM suggestions"
        )
        params: tuple = ()
        if status:
            q += " WHERE status=?"
            params = (status,)
        return pd.read_sql_query(
            q + " ORDER BY forecast_origin, stock_code", self.conn, params=params
        )

    def _log(self, suggestion_id, action, actor, note, at) -> None:
        self.conn.execute(
            "INSERT INTO audit_log (suggestion_id, action, actor, note, at) VALUES (?,?,?,?,?)",
            (suggestion_id, action, actor, note, at),
        )

    def close(self) -> None:
        self.conn.close()


def _jsonable(v):
    if v is None or (isinstance(v, float) and v != v) or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    if hasattr(v, "item"):
        return v.item()
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Approve or reject reorder suggestions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register", help="register suggestions from a stockout-risk file")
    reg.add_argument("risk_file", type=Path)
    lst = sub.add_parser("list")
    lst.add_argument("--status", default="pending_approval")
    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("suggestion_id")
        p.add_argument("--by", required=True, help="name of the person deciding")
        p.add_argument("--note")
    show = sub.add_parser("show")
    show.add_argument("suggestion_id")
    args = parser.parse_args(argv)

    store = ApprovalStore()
    try:
        if args.cmd == "register":
            n = store.register(pd.read_parquet(args.risk_file))
            print(f"registered {n} new suggestions as pending_approval")
        elif args.cmd == "list":
            print(store.list(args.status or None).to_string(index=False))
        elif args.cmd == "show":
            print(json.dumps(store.get(args.suggestion_id), indent=2))
        else:
            result = store.decide(args.suggestion_id, COMMANDS[args.cmd], args.by, args.note)
            print(json.dumps(result, indent=2))
    except ApprovalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
