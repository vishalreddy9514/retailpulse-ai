"""Stream processor tests: the streaming path must reproduce the batch aggregates exactly."""

from __future__ import annotations

import zlib

import pandas as pd
import pytest

from src.processing.aggregate_sales import aggregate
from src.processing.clean_transactions import clean_transactions
from src.streaming.reconcile import load_stream, reconcile
from src.streaming.replay import InMemorySink, ReplayConfig, prepare, replay
from src.streaming.schema import TOPIC_DLQ, TOPIC_TRANSACTIONS
from src.streaming.stream_processor import Message, StreamProcessor

PARTITIONS = 3

ROWS = [
    # Day 1 ---------------------------------------------------------------------------
    ("500001", "11111", "MUG", 10, "2011-03-07 09:00", 2.0, 1.0, "UK"),
    ("500001", "11111", "MUG", 10, "2011-03-07 09:00", 2.0, 1.0, "UK"),  # exact duplicate
    ("500002", "11111", "MUG", 5, "2011-03-07 09:30", 2.0, None, "UK"),  # no customer
    # Two identical sales then one cancellation: the later sale is reversed.
    ("500003", "22222", "BOWL", 4, "2011-03-07 10:00", 3.0, 7.0, "UK"),
    ("500004", "22222", "BOWL", 4, "2011-03-07 10:30", 3.0, 7.0, "UK"),
    ("C500005", "22222", "BOWL", -4, "2011-03-07 11:00", 3.0, 7.0, "UK"),
    # An ordinary sale first, so this product-day is inside the batch table's window...
    ("500016", "33333", "JUG", 3, "2011-03-07 11:30", 5.0, 5.0, "UK"),
    # ...then, in the same minute, a cancellation replayed BEFORE the sale it reverses.
    ("C500006", "33333", "JUG", -2, "2011-03-07 12:00", 5.0, 9.0, "UK"),
    ("500007", "33333", "JUG", 2, "2011-03-07 12:00", 5.0, 9.0, "UK"),
    # Cancellation without a customer id is never matched.
    ("500008", "44444", "VASE", 1, "2011-03-07 13:00", 9.0, None, "UK"),
    ("C500009", "44444", "VASE", -1, "2011-03-07 13:05", 9.0, None, "UK"),
    ("500010", "POST", "POSTAGE", 1, "2011-03-07 14:00", 18.0, 1.0, "UK"),
    ("500011", "11111", "damaged", -3, "2011-03-07 15:00", 0.0, None, "UK"),
    # Day 2: an ordinary return of an earlier sale, more sales.
    ("C500012", "11111", "MUG", -2, "2011-03-08 09:00", 2.0, 1.0, "UK"),
    ("500013", "11111", "MUG", 7, "2011-03-08 10:00", 2.5, 2.0, "UK"),
    ("500014", "22222", "BOWL", 1, "2011-03-08 17:00", 3.0, 8.0, "UK"),
    ("500015", "33333", "JUG", 1, "2011-03-08 17:00", 5.0, 8.0, "UK"),
]


def kafka_messages(raw: pd.DataFrame, start_offsets: dict[int, int] | None = None) -> list[Message]:
    """Replay through the real engine, then lay events out over partitions as Kafka would."""
    sink = InMemorySink()
    replay(prepare(raw), sink, ReplayConfig(from_start=True))
    offsets = dict(start_offsets or {})
    out = []
    for topic, key, value, _ in sink.committed:
        p = zlib.crc32(key) % PARTITIONS
        off = offsets.get((topic, p), 0)
        offsets[(topic, p)] = off + 1
        out.append(Message(topic, p, off, value))
    return out


def batch_daily(raw: pd.DataFrame) -> pd.DataFrame:
    clean, _ = clean_transactions(raw)
    tables, _ = aggregate(clean)
    return tables["product_daily"]


@pytest.fixture
def raw(make_raw):
    return make_raw(ROWS)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "state.sqlite"


def run_stream(db, messages, batch_size, finalize=True):
    proc = StreamProcessor(db, dead_letter_topic=TOPIC_DLQ)
    for i in range(0, len(messages), batch_size):
        proc.process(messages[i : i + batch_size])
    if finalize:
        proc.finalize()
    return proc


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_stream_matches_batch_exactly(raw, db, batch_size):
    msgs = kafka_messages(raw)
    run_stream(db, msgs, batch_size).close()
    report = reconcile(load_stream(db), batch_daily(raw))
    assert report["exact_match"], report
    assert report["product_days_compared"] == 7
    assert report["stream_rows_before_first_sale"]["rows"] == 0  # every row is compared


def test_reversal_semantics(raw, db):
    proc = run_stream(db, kafka_messages(raw), 100)
    types = dict(proc.conn.execute("SELECT invoice, line_type FROM events"))
    assert types["500003"] == "sale" and types["500004"] == "order_reversal_sale"
    assert types["C500005"] == "order_reversal_return"
    assert types["500007"] == "order_reversal_sale" and types["C500006"] == "order_reversal_return"
    assert types["C500009"] == "cancellation_return"
    counters = proc.status()["counters"]
    assert counters["duplicates_ignored"] == 1
    assert counters["reversals_matched"] == 2
    proc.close()


def test_pending_cancellation_waits_for_the_minute_to_close(raw, db):
    msgs = kafka_messages(raw)
    cancel = next(i for i, m in enumerate(msgs) if b'"C500006"' in m.value)
    proc = StreamProcessor(db, dead_letter_topic=TOPIC_DLQ)
    proc.process(msgs[: cancel + 1])
    pending = proc.conn.execute("SELECT reversal_pending FROM events WHERE invoice='C500006'")
    assert pending.fetchone()[0] == 1  # its sale (same minute) has not arrived yet
    row = proc.conn.execute(
        "SELECT units_returned FROM product_daily WHERE stock_code='33333' AND date='2011-03-07'"
    ).fetchone()
    assert row[0] == 2  # provisionally a return
    proc.process(msgs[cancel + 1 :])
    proc.finalize()
    row = proc.conn.execute(
        "SELECT units_sold, units_returned, units_reversed FROM product_daily"
        " WHERE stock_code='33333' AND date='2011-03-07'"
    ).fetchone()
    assert row == (3, 0, 2)  # the ordinary 3-unit sale remains; 2 units reversed
    proc.close()


def test_redelivered_messages_are_skipped_by_stored_offsets(raw, db):
    msgs = kafka_messages(raw)
    proc = run_stream(db, msgs, 100)
    before = proc.conn.execute("SELECT * FROM product_daily ORDER BY 1, 2").fetchall()
    result = proc.process(msgs)  # broker redelivers everything (e.g. lost group commit)
    assert result.already_processed == len(msgs) and result.applied == 0
    assert proc.conn.execute("SELECT * FROM product_daily ORDER BY 1, 2").fetchall() == before
    proc.close()


def test_replay_from_start_is_deduplicated_by_event_id(raw, db):
    first = kafka_messages(raw)
    end = {}
    for m in first:
        end[(m.topic, m.partition)] = m.offset + 1
    second = kafka_messages(raw, start_offsets=end)  # same events at new offsets
    proc = run_stream(db, first + second, 4)
    assert proc.status()["counters"]["duplicates_ignored"] == 1 + len(second)
    proc.close()
    assert reconcile(load_stream(db), batch_daily(raw))["exact_match"]


def test_crash_mid_batch_rolls_back_everything(raw, db, monkeypatch):
    msgs = kafka_messages(raw)
    proc = StreamProcessor(db, dead_letter_topic=TOPIC_DLQ)
    proc.process(msgs[:4])
    snapshot = (
        proc.conn.execute("SELECT * FROM product_daily ORDER BY 1, 2").fetchall(),
        proc.stored_offsets(),
        proc.conn.execute("SELECT COUNT(*) FROM events").fetchone(),
    )
    calls = {"n": 0}
    original = proc._apply

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(proc, "_apply", flaky)
    with pytest.raises(RuntimeError):
        proc.process(msgs[4:])
    assert (
        proc.conn.execute("SELECT * FROM product_daily ORDER BY 1, 2").fetchall(),
        proc.stored_offsets(),
        proc.conn.execute("SELECT COUNT(*) FROM events").fetchone(),
    ) == snapshot
    monkeypatch.setattr(proc, "_apply", original)
    proc.process(msgs[4:])
    proc.finalize()
    proc.close()
    assert reconcile(load_stream(db), batch_daily(raw))["exact_match"]


def test_poison_messages_go_to_dead_letters_and_offsets_advance(db):
    proc = StreamProcessor(db, dead_letter_topic=TOPIC_DLQ)
    result = proc.process(
        [
            Message(TOPIC_TRANSACTIONS, 0, 0, b'{"not": "an event"}'),
            Message(TOPIC_DLQ, 1, 0, b'{"errors": []}'),
        ]
    )
    assert result.dead_letters == 2 and result.applied == 0
    assert proc.stored_offsets() == {(TOPIC_TRANSACTIONS, 0): 1, (TOPIC_DLQ, 1): 1}
    assert proc.conn.execute("SELECT COUNT(*) FROM dead_letters").fetchone()[0] == 2
    proc.close()


def test_status_reports_latency_and_origin(raw, db):
    proc = run_stream(db, kafka_messages(raw), 100)
    status = proc.status()
    assert status["data_origin"].startswith("historical_replay")
    assert status["latency_ms_ingestion_to_processed"]["p50"] is not None
    assert status["pending_reversal_checks"] == 0
    proc.close()
