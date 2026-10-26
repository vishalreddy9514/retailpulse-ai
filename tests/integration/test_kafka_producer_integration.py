"""Replay against a real Kafka broker.

Skipped unless RETAILPULSE_KAFKA_BOOTSTRAP is set, for example:

    docker compose up -d kafka
    RETAILPULSE_KAFKA_BOOTSTRAP=localhost:9092 pytest -m integration
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("RETAILPULSE_KAFKA_BOOTSTRAP")
if not BOOTSTRAP:
    pytest.skip("RETAILPULSE_KAFKA_BOOTSTRAP not set", allow_module_level=True)

from confluent_kafka import Consumer, KafkaError  # noqa: E402

from src.streaming.kafka_producer import KafkaSink, ensure_topics  # noqa: E402
from src.streaming.replay import ReplayConfig, prepare, replay  # noqa: E402
from src.streaming.schema import TransactionEvent  # noqa: E402


class CrashingKafkaSink(KafkaSink):
    def __init__(self, *args, crash_after: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.crash_after = crash_after
        self.sent = 0

    def send(self, topic, key, value, headers):
        self.sent += 1
        if self.crash_after is not None and self.sent > self.crash_after:
            raise RuntimeError("simulated crash")
        super().send(topic, key, value, headers)


def read_committed(topics: list[str]) -> list:
    c = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"it-{uuid.uuid4().hex[:8]}",
            "isolation.level": "read_committed",
            "auto.offset.reset": "earliest",
            "enable.partition.eof": True,
        }
    )
    c.subscribe(topics)
    out, eof = [], set()
    while True:
        msg = c.poll(10)
        assert msg is not None, "timed out waiting for messages"
        if msg.error():
            assert msg.error().code() == KafkaError._PARTITION_EOF, msg.error()
            eof.add((msg.topic(), msg.partition()))
            if len(eof) == 6 * len(topics):
                break
            continue
        out.append(msg)
    c.close()
    return out


def test_crash_and_resume_delivers_every_event_exactly_once(make_raw):
    run = uuid.uuid4().hex[:8]
    topic, dlq = f"it.transactions.{run}", f"it.dlq.{run}"
    ensure_topics(BOOTSTRAP, [topic, dlq])
    rows = [
        (
            f"5{i:05d}",
            f"2{i % 7:04d}",
            "ITEM",
            1 + i % 3,
            f"2011-03-01 {8 + i // 60:02d}:{i % 60:02d}",
            1.0,
            1.0,
            "UK",
        )
        for i in range(250)
    ]
    rows.append(("BAD", "22222", "ITEM", 1, "2011-03-01 18:00", 1.0, 1.0, "UK"))  # -> DLQ
    frame = prepare(make_raw(rows))
    config = ReplayConfig(batch_size=40, topic=topic, dlq_topic=dlq)
    txid = f"it-replay-{run}"

    crashing = CrashingKafkaSink(BOOTSTRAP, topic, dlq, transactional_id=txid, crash_after=130)
    with pytest.raises(RuntimeError, match="simulated crash"):
        replay(frame, crashing, config)
    assert crashing.last_committed_sequence() == 119  # three batches of 40 committed

    sink = KafkaSink(BOOTSTRAP, topic, dlq, transactional_id=txid)
    stats = replay(frame, sink, config)
    assert stats.resumed_after_sequence == 119
    assert stats.published + stats.rejected == 131

    messages = read_committed([topic, dlq])
    events = [TransactionEvent.from_bytes(m.value()) for m in messages if m.topic() == topic]
    dead = [m for m in messages if m.topic() == dlq]
    assert sorted(e.sequence for e in events) == list(range(250))  # aborted sends never visible
    assert len(dead) == 1
    assert all(
        m.key().decode() == TransactionEvent.from_bytes(m.value()).stock_code
        for m in messages
        if m.topic() == topic
    )

    # A further rerun publishes nothing.
    again = replay(frame, KafkaSink(BOOTSTRAP, topic, dlq, transactional_id=txid), config)
    assert again.published == 0 and again.rejected == 0
