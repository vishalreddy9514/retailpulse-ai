"""Replay engine tests with an in-memory transactional sink and a fake clock (no broker)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.processing.clean_transactions import clean_transactions
from src.streaming.replay import InMemorySink, ReplayConfig, prepare, replay, window
from src.streaming.schema import TOPIC_DLQ, TOPIC_TRANSACTIONS, TransactionEvent


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def utcnow(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self.t)


ROWS = [
    # Deliberately out of order: replay must sort by event time.
    ("500003", "22222", "MUG", 1, "2011-03-01 10:10", 1.0, 3.0, "UK"),
    ("500001", "85123A", "HEART", 6, "2011-03-01 10:00", 2.55, 1.0, "UK"),
    ("C500002", "85123A", "HEART", -6, "2011-03-01 10:05", 2.55, 1.0, "UK"),
    ("500004", "POST", "POSTAGE", 1, "2011-03-01 12:10", 18.0, None, "France"),
    ("500005", "22222", "MUG", 2, "2011-03-02 09:00", 1.0, None, "UK"),
]


@pytest.fixture
def frame(make_raw):
    return prepare(make_raw(ROWS))


def _events(sink: InMemorySink, topic=TOPIC_TRANSACTIONS) -> list[TransactionEvent]:
    return [TransactionEvent.from_bytes(v) for t, _, v, _ in sink.committed if t == topic]


def test_events_are_replayed_in_event_time_order(frame):
    sink = InMemorySink()
    stats = replay(frame, sink, ReplayConfig(), FakeClock())
    events = _events(sink)
    assert [e.invoice for e in events] == ["500001", "C500002", "500003", "500004", "500005"]
    assert [e.sequence for e in events] == [0, 1, 2, 3, 4]
    assert stats.published == 5 and stats.rejected == 0


def test_event_payload_preserves_event_time_and_labels_origin(frame):
    sink = InMemorySink()
    clock = FakeClock()
    replay(frame, sink, ReplayConfig(), clock)
    first = _events(sink)[0]
    assert first.event_time == datetime(2011, 3, 1, 10, 0)  # original timestamp, unchanged
    assert first.ingestion_time.tzinfo is not None  # wall-clock UTC
    assert first.data_origin == "historical_replay"
    assert first.quantity == 6 and first.unit_price == 2.55 and first.customer_id == "1"
    topic, key, _, headers = sink.committed[0]
    assert key == b"85123A"  # keyed by product: per-product ordering within a partition
    assert headers["sequence"] == b"0" and headers["data_origin"] == b"historical_replay"


def test_event_ids_match_the_batch_pipeline(make_raw, frame):
    sink = InMemorySink()
    replay(frame, sink, ReplayConfig(), FakeClock())
    clean, _ = clean_transactions(make_raw(ROWS))
    assert {e.event_id for e in _events(sink)} == set(clean["event_id"])


def test_malformed_records_go_to_dead_letter_topic(make_raw):
    rows = ROWS + [
        ("500006", "22222", "MUG", 1, "2011-03-02 10:00", float("nan"), 1.0, "UK"),  # no price
        ("XX", "22222", "MUG", 1, "2011-03-02 11:00", 1.0, 1.0, "UK"),  # bad invoice
        ("500008", "22222", "MUG", None, "2011-03-02 12:00", 1.0, 1.0, "UK"),  # no quantity
    ]
    sink = InMemorySink()
    stats = replay(prepare(make_raw(rows)), sink, ReplayConfig(), FakeClock())
    assert stats.published == 5 and stats.rejected == 3
    dlq = [json.loads(v) for t, _, v, _ in sink.committed if t == TOPIC_DLQ]
    fields = sorted(e["field"] for d in dlq for e in d["errors"])
    assert fields == ["invoice", "quantity", "unit_price"]
    assert all(d["data_origin"] == "historical_replay" and "raw" in d for d in dlq)
    assert {e.invoice for e in _events(sink)}.isdisjoint({"500006", "XX", "500008"})


def test_restart_after_crash_has_no_duplicates_or_gaps(make_raw):
    rows = [
        (f"5{i:05d}", "22222", "MUG", 1, f"2011-03-01 10:{i:02d}", 1.0, 1.0, "UK")
        for i in range(50)
    ]
    frame = prepare(make_raw(rows))
    sink = InMemorySink(fail_after_sends=23)
    with pytest.raises(RuntimeError, match="simulated crash"):
        replay(frame, sink, ReplayConfig(batch_size=10), FakeClock())
    assert sink.aborts == 1
    assert sink.last_committed_sequence() == 19  # two full batches committed; third aborted

    sink.fail_after_sends = None
    stats = replay(frame, sink, ReplayConfig(batch_size=10), FakeClock())
    assert stats.resumed_after_sequence == 19
    seqs = [e.sequence for e in _events(sink)]
    assert seqs == list(range(50))  # every event exactly once, in order


def test_rerun_after_completion_publishes_nothing(frame):
    sink = InMemorySink()
    replay(frame, sink, ReplayConfig(), FakeClock())
    stats = replay(frame, sink, ReplayConfig(), FakeClock())
    assert stats.published == 0 and stats.resumed_after_sequence == 4


def test_from_start_replays_again_on_purpose(frame):
    sink = InMemorySink()
    replay(frame, sink, ReplayConfig(), FakeClock())
    stats = replay(frame, sink, ReplayConfig(from_start=True), FakeClock())
    assert stats.published == 5
    ids = [e.event_id for e in _events(sink)]
    assert len(ids) == 10 and len(set(ids)) == 5  # same ids: the consumer can de-duplicate


def test_accelerated_mode_scales_and_caps_gaps(frame):
    clock = FakeClock()
    replay(
        frame, InMemorySink(), ReplayConfig(mode="accelerated", speed=60, max_gap_seconds=30), clock
    )
    # Gaps: 5 min, 5 min, 2 h, ~21 h -> 5 s, 5 s, then capped at 30 s twice.
    assert clock.sleeps == pytest.approx([5.0, 5.0, 30.0, 30.0])


def test_fast_mode_does_not_sleep_unless_rate_capped(frame):
    clock = FakeClock()
    replay(frame, InMemorySink(), ReplayConfig(mode="fast"), clock)
    assert clock.sleeps == []
    capped = FakeClock()
    replay(frame, InMemorySink(), ReplayConfig(mode="fast", max_events_per_second=2), capped)
    assert sum(capped.sleeps) == pytest.approx(2.5)  # 5 events at 2/s


def test_open_batch_is_committed_before_a_long_wait(frame):
    sink = InMemorySink()
    clock = FakeClock()
    replay(frame, sink, ReplayConfig(mode="accelerated", speed=60, batch_size=1000), clock)
    # Each of the 4 waits commits the events sent so far, then the final commit.
    assert sink.commits == 5


def test_window_selects_by_date_and_limit(frame):
    assert len(window(frame, start="2011-03-02")) == 1
    assert len(window(frame, end="2011-03-01")) == 4
    assert len(window(frame, limit=2)) == 2


def test_config_validation():
    with pytest.raises(ValueError):
        ReplayConfig(mode="live")
    with pytest.raises(ValueError):
        ReplayConfig(speed=0)
    with pytest.raises(ValueError):
        ReplayConfig(max_events_per_second=-1)


def test_schema_rejects_timezone_on_event_time_and_naive_ingestion_time():
    base = dict(
        event_id="a" * 32,
        sequence=0,
        replay_run_id="r",
        invoice="500001",
        stock_code="22222",
        quantity=1,
        unit_price=1.0,
        country="UK",
        source_sheet="s",
        source_row=2,
    )
    with pytest.raises(ValueError):
        TransactionEvent(
            **base, event_time=datetime(2011, 1, 1, tzinfo=UTC), ingestion_time=datetime.now(UTC)
        )
    with pytest.raises(ValueError):
        TransactionEvent(
            **base, event_time=datetime(2011, 1, 1), ingestion_time=datetime(2026, 1, 1)
        )
    with pytest.raises(ValueError):
        TransactionEvent(
            **base, event_time=datetime(2011, 1, 1), ingestion_time=datetime.now(UTC), extra=1
        )


def test_committed_json_schema_matches_the_model():
    from pathlib import Path

    from src.streaming.schema import json_schema

    path = Path(__file__).resolve().parents[2] / "schemas" / "transaction_event.v1.json"
    assert json.loads(path.read_text()) == json_schema(), (
        "Event schema changed: regenerate schemas/transaction_event.v1.json "
        "(and bump the schema version if the change is breaking)"
    )
