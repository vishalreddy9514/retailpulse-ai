"""Stateful, idempotent processing of replayed transaction events into daily demand.

The state lives in one SQLite database (local-first, no extra service):

``events``
    De-duplicated event store, keyed by ``event_id``. A replayed duplicate, or one of
    the raw file's exact-duplicate lines, is ignored, as the batch cleaning does.
``product_daily``
    Incrementally maintained product x day counters, matching the batch
    ``product_daily`` columns. Distinct invoice and customer counts come from
    ``product_day_invoices`` / ``product_day_customers``.
``offsets``
    The next Kafka offset per partition, written in the *same* SQLite transaction as the
    events and aggregates they cover. A crash at any point either keeps a whole poll
    batch, including its offsets, or none of it. On restart the consumer seeks to these
    offsets, so every event is applied exactly once.

Classification uses the batch rules (``classify_line``). Same-day order reversals are
matched exactly as the batch does: a cancellation first counts as a return. Its match is
resolved once the partition's event time has moved past the cancellation's minute, so
every sale with a timestamp up to that minute has arrived. Products are partitioned by
stock code, and the match key includes stock code, so this per-partition watermark is
enough. The match then uses the batch's greedy one-to-one rule and moves the units from
returns/sales into ``units_reversed``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from src.processing.clean_transactions import (
    CANCELLATION_RETURN,
    ORDER_REVERSAL_RETURN,
    ORDER_REVERSAL_SALE,
    SALE,
    classify_line,
)
from src.streaming.schema import TransactionEvent

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL,
    kafka_topic TEXT, kafka_partition INTEGER, kafka_offset INTEGER,
    invoice TEXT NOT NULL, stock_code TEXT NOT NULL, description TEXT,
    quantity INTEGER NOT NULL, unit_price REAL NOT NULL, line_revenue REAL NOT NULL,
    customer_id TEXT, country TEXT NOT NULL,
    event_time TEXT NOT NULL, invoice_day TEXT NOT NULL,
    ingestion_time TEXT NOT NULL, processed_at TEXT NOT NULL, latency_ms REAL NOT NULL,
    line_type TEXT NOT NULL,
    reversal_pending INTEGER NOT NULL DEFAULT 0,
    matched_event_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_match
    ON events(stock_code, customer_id, invoice_day, quantity, line_type);
CREATE INDEX IF NOT EXISTS ix_events_pending ON events(reversal_pending, event_time, sequence);
CREATE INDEX IF NOT EXISTS ix_events_day ON events(invoice_day);
CREATE TABLE IF NOT EXISTS product_daily (
    stock_code TEXT NOT NULL, date TEXT NOT NULL,
    units_sold INTEGER NOT NULL DEFAULT 0, sale_lines INTEGER NOT NULL DEFAULT 0,
    gross_revenue REAL NOT NULL DEFAULT 0,
    units_returned INTEGER NOT NULL DEFAULT 0, return_lines INTEGER NOT NULL DEFAULT 0,
    return_value REAL NOT NULL DEFAULT 0,
    units_reversed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (stock_code, date)
);
CREATE TABLE IF NOT EXISTS product_day_invoices (
    stock_code TEXT, date TEXT, invoice TEXT, sale_lines INTEGER NOT NULL,
    PRIMARY KEY (stock_code, date, invoice)
);
CREATE TABLE IF NOT EXISTS product_day_customers (
    stock_code TEXT, date TEXT, customer_id TEXT, sale_lines INTEGER NOT NULL,
    PRIMARY KEY (stock_code, date, customer_id)
);
CREATE TABLE IF NOT EXISTS offsets (
    topic TEXT, partition INTEGER, next_offset INTEGER NOT NULL,
    PRIMARY KEY (topic, partition)
);
CREATE TABLE IF NOT EXISTS partition_watermarks (
    topic TEXT, partition INTEGER, max_event_time TEXT NOT NULL,
    PRIMARY KEY (topic, partition)
);
CREATE TABLE IF NOT EXISTS dead_letters (
    topic TEXT, partition INTEGER, offset INTEGER, reason TEXT, payload BLOB, received_at TEXT,
    PRIMARY KEY (topic, partition, offset)
);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE VIEW IF NOT EXISTS product_daily_view AS
SELECT d.*,
       d.units_sold - d.units_returned AS net_units,
       ROUND(d.gross_revenue - d.return_value, 2) AS net_revenue,
       (SELECT COUNT(*) FROM product_day_invoices i
         WHERE i.stock_code = d.stock_code AND i.date = d.date AND i.sale_lines > 0) AS invoices,
       (SELECT COUNT(*) FROM product_day_customers c
         WHERE c.stock_code = d.stock_code AND c.date = d.date AND c.sale_lines > 0) AS customers
FROM product_daily d;
"""


EVENT_COLUMNS = (
    "event_id", "sequence", "kafka_topic", "kafka_partition", "kafka_offset",
    "invoice", "stock_code", "description", "quantity", "unit_price", "line_revenue",
    "customer_id", "country", "event_time", "invoice_day",
    "ingestion_time", "processed_at", "latency_ms", "line_type", "reversal_pending",
)  # fmt: skip


@dataclass(frozen=True)
class Message:
    """The parts of a Kafka message the processor needs (decoupled from the client library)."""

    topic: str
    partition: int
    offset: int
    value: bytes


@dataclass
class BatchResult:
    applied: int = 0
    duplicates: int = 0
    dead_letters: int = 0
    reversals_matched: int = 0
    already_processed: int = 0


class StreamProcessor:
    def __init__(self, db_path: Path | str, dead_letter_topic: str | None = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.dead_letter_topic = dead_letter_topic

    # -- offsets ---------------------------------------------------------------------------
    def stored_offsets(self) -> dict[tuple[str, int], int]:
        rows = self.conn.execute("SELECT topic, partition, next_offset FROM offsets")
        return {(t, p): o for t, p, o in rows}

    # -- processing ------------------------------------------------------------------------
    def process(self, messages: Iterable[Message], now: datetime | None = None) -> BatchResult:
        """Apply a poll batch atomically: events, aggregates, reversals and offsets together."""
        result = BatchResult()
        stored = self.stored_offsets()
        touched: set[tuple[str, int]] = set()
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            for msg in messages:
                key = (msg.topic, msg.partition)
                if msg.offset < stored.get(key, 0):
                    result.already_processed += 1  # redelivered below the stored offset
                    continue
                if msg.topic == self.dead_letter_topic:
                    self._dead_letter(msg, "producer dead-letter record")
                    result.dead_letters += 1
                else:
                    self._apply(msg, now or datetime.now(UTC), result)
                stored[key] = msg.offset + 1
                c.execute(
                    "INSERT INTO offsets VALUES (?,?,?) "
                    "ON CONFLICT(topic, partition) DO UPDATE SET next_offset=excluded.next_offset",
                    (msg.topic, msg.partition, msg.offset + 1),
                )
                touched.add(key)
            result.reversals_matched += self.resolve_pending(final=False)
            self._bump("events_applied", result.applied)
            self._bump("duplicates_ignored", result.duplicates)
            self._bump("dead_letters", result.dead_letters)
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        return result

    def _apply(self, msg: Message, now: datetime, result: BatchResult) -> None:
        try:
            ev = TransactionEvent.from_bytes(msg.value)
        except ValidationError as exc:
            self._dead_letter(msg, f"schema: {exc.error_count()} error(s)")
            result.dead_letters += 1
            return
        line_type = classify_line(ev.invoice, ev.stock_code, ev.quantity, ev.unit_price)
        day = ev.event_time.date().isoformat()
        revenue = round(ev.quantity * ev.unit_price, 2)
        latency_ms = (now - ev.ingestion_time).total_seconds() * 1000
        pending = int(line_type == CANCELLATION_RETURN and ev.customer_id is not None)
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO events ({', '.join(EVENT_COLUMNS)})"
            f" VALUES ({', '.join('?' * len(EVENT_COLUMNS))})",
            (
                ev.event_id, ev.sequence, msg.topic, msg.partition, msg.offset,
                ev.invoice, ev.stock_code, ev.description, ev.quantity, ev.unit_price, revenue,
                ev.customer_id, ev.country, ev.event_time.isoformat(sep=" "), day,
                ev.ingestion_time.isoformat(), now.isoformat(), latency_ms,
                line_type, pending,
            ),
        )  # fmt: skip
        if cur.rowcount == 0:
            result.duplicates += 1
            return
        result.applied += 1
        self.conn.execute(
            "INSERT INTO partition_watermarks VALUES (?,?,?) ON CONFLICT(topic, partition) "
            "DO UPDATE SET max_event_time=MAX(max_event_time, excluded.max_event_time)",
            (msg.topic, msg.partition, ev.event_time.isoformat(sep=" ")),
        )
        if line_type == SALE:
            self._sale(ev.stock_code, day, ev.invoice, ev.customer_id, ev.quantity, revenue, +1)
        elif line_type == CANCELLATION_RETURN:
            self._return(ev.stock_code, day, -ev.quantity, -revenue, +1)

    def resolve_pending(self, final: bool = False) -> int:
        """Match pending cancellations whose minute is complete (all of them if ``final``)."""
        c = self.conn
        pending = c.execute(
            "SELECT e.event_id, e.stock_code, e.customer_id, e.invoice_day, e.quantity,"
            " e.event_time, e.line_revenue, w.max_event_time"
            " FROM events e JOIN partition_watermarks w"
            " ON w.topic = e.kafka_topic AND w.partition = e.kafka_partition"
            " WHERE e.reversal_pending = 1 ORDER BY e.event_time, e.sequence"
        ).fetchall()
        matched = 0
        for event_id, code, cust, day, qty, ts, ret_revenue, watermark in pending:
            if not final and ts >= watermark:
                continue  # more events with this timestamp may still arrive on the partition
            sale = c.execute(
                "SELECT event_id, invoice, line_revenue FROM events"
                " WHERE stock_code=? AND customer_id=? AND invoice_day=? AND quantity=?"
                " AND line_type=? AND event_time <= ?"
                " ORDER BY event_time DESC, sequence DESC LIMIT 1",
                (code, cust, day, -qty, SALE, ts),
            ).fetchone()
            if sale is None:
                c.execute("UPDATE events SET reversal_pending=0 WHERE event_id=?", (event_id,))
                continue
            sale_id, sale_invoice, sale_revenue = sale
            c.execute(
                "UPDATE events SET line_type=?, reversal_pending=0, matched_event_id=?"
                " WHERE event_id=?",
                (ORDER_REVERSAL_RETURN, sale_id, event_id),
            )
            c.execute(
                "UPDATE events SET line_type=?, matched_event_id=? WHERE event_id=?",
                (ORDER_REVERSAL_SALE, event_id, sale_id),
            )
            # Undo the provisional return and the sale; record the reversed units instead.
            self._return(code, day, -qty, -ret_revenue, -1)
            self._sale(code, day, sale_invoice, cust, -qty, sale_revenue, -1)
            c.execute(
                "UPDATE product_daily SET units_reversed = units_reversed + ?"
                " WHERE stock_code=? AND date=?",
                (-qty, code, day),
            )
            matched += 1
        self._bump("reversals_matched", matched)
        return matched

    def finalize(self) -> int:
        """End of a replay window: no more events will arrive, so resolve everything pending."""
        c = self.conn
        c.execute("BEGIN IMMEDIATE")
        try:
            n = self.resolve_pending(final=True)
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        return n

    # -- aggregate helpers -----------------------------------------------------------------
    def _ensure_row(self, code: str, day: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO product_daily (stock_code, date) VALUES (?, ?)", (code, day)
        )

    def _sale(self, code, day, invoice, customer, qty, revenue, sign) -> None:
        self._ensure_row(code, day)
        self.conn.execute(
            "UPDATE product_daily SET units_sold = units_sold + ?, sale_lines = sale_lines + ?,"
            " gross_revenue = gross_revenue + ? WHERE stock_code=? AND date=?",
            (sign * qty, sign, sign * revenue, code, day),
        )
        self.conn.execute(
            "INSERT INTO product_day_invoices VALUES (?,?,?,?) ON CONFLICT DO UPDATE"
            " SET sale_lines = sale_lines + excluded.sale_lines",
            (code, day, invoice, sign),
        )
        if customer is not None:
            self.conn.execute(
                "INSERT INTO product_day_customers VALUES (?,?,?,?) ON CONFLICT DO UPDATE"
                " SET sale_lines = sale_lines + excluded.sale_lines",
                (code, day, customer, sign),
            )

    def _return(self, code, day, units, value, sign) -> None:
        self._ensure_row(code, day)
        self.conn.execute(
            "UPDATE product_daily SET units_returned = units_returned + ?,"
            " return_lines = return_lines + ?, return_value = return_value + ?"
            " WHERE stock_code=? AND date=?",
            (sign * units, sign, sign * value, code, day),
        )

    def _dead_letter(self, msg: Message, reason: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO dead_letters VALUES (?,?,?,?,?,?)",
            (
                msg.topic,
                msg.partition,
                msg.offset,
                reason,
                msg.value,
                datetime.now(UTC).isoformat(),
            ),
        )

    def _bump(self, name: str, by: int) -> None:
        if by:
            self.conn.execute(
                "INSERT INTO counters VALUES (?, ?)"
                " ON CONFLICT(name) DO UPDATE SET value = value + ?",
                (name, by, by),
            )

    # -- reporting -------------------------------------------------------------------------
    def status(self) -> dict:
        c = self.conn
        counters = dict(c.execute("SELECT name, value FROM counters"))
        row = c.execute(
            "SELECT COUNT(*), MIN(event_time), MAX(event_time), MAX(processed_at) FROM events"
        ).fetchone()
        lat = [r[0] for r in c.execute("SELECT latency_ms FROM events ORDER BY latency_ms")]

        def pct(p: float) -> float | None:
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None

        return {
            "data_origin": "historical_replay (simulated real-time)",
            "events_stored": row[0],
            "event_time_range": [row[1], row[2]],
            "last_processed_at": row[3],
            "pending_reversal_checks": c.execute(
                "SELECT COUNT(*) FROM events WHERE reversal_pending=1"
            ).fetchone()[0],
            "counters": counters,
            "latency_ms_ingestion_to_processed": {
                "p50": pct(0.5),
                "p95": pct(0.95),
                "max": pct(1.0),
            },
            "line_types": dict(c.execute("SELECT line_type, COUNT(*) FROM events GROUP BY 1")),
            "offsets": {f"{t}[{p}]": o for (t, p), o in self.stored_offsets().items()},
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> StreamProcessor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
