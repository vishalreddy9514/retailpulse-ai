"""Kafka consumer: replayed transaction events -> de-duplicated store and daily demand.

SIMULATED REAL-TIME. The input is the historical replay from ``kafka_producer.py``, not
live retailer data.

* Reads with ``isolation.level=read_committed``, so aborted producer batches are never
  seen.
* Partitions are assigned explicitly and positioned at the offsets stored in the SQLite
  state database, which are written atomically with the results. Restarting continues
  exactly where the last committed poll batch ended. Kafka consumer-group offsets are
  also committed, but only so standard lag tooling works; SQLite is the source of truth.
* Events are de-duplicated on ``event_id``, so re-running the producer with
  ``--from-start`` changes nothing.

Usage::

    python -m src.streaming.kafka_consumer                        # run until Ctrl-C
    python -m src.streaming.kafka_consumer --idle-exit 10         # catch up, then stop
    python -m src.streaming.kafka_consumer --status               # print state and latency
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from confluent_kafka import OFFSET_BEGINNING, Consumer, KafkaError, KafkaException, TopicPartition
from prometheus_client import start_http_server

from src import config
from src import observability as obs
from src.streaming.kafka_producer import ensure_topics
from src.streaming.schema import TOPIC_DLQ, TOPIC_TRANSACTIONS
from src.streaming.stream_processor import Message, StreamProcessor

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP = os.environ.get("RETAILPULSE_KAFKA_BOOTSTRAP", "localhost:9092")
DEFAULT_DB = config.DATA_DIR / "stream" / "stream_state.sqlite"
GROUP_ID = "retailpulse-stream-processor"


def make_consumer(bootstrap: str, group_id: str = GROUP_ID) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "isolation.level": "read_committed",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "auto.offset.reset": "earliest",
        }
    )


def assign_from_store(consumer: Consumer, processor: StreamProcessor, topics: list[str]) -> list:
    stored = processor.stored_offsets()
    assignment = []
    for topic in topics:
        meta = consumer.list_topics(topic, timeout=10).topics[topic]
        if meta.error is not None:
            raise KafkaException(meta.error)
        for p in sorted(meta.partitions):
            assignment.append(TopicPartition(topic, p, stored.get((topic, p), OFFSET_BEGINNING)))
    consumer.assign(assignment)
    return assignment


def record_metrics(result, seconds: float, watermark: str | None) -> None:
    for name in ("applied", "duplicates", "dead_letters", "already_processed"):
        if getattr(result, name):
            obs.STREAM_MESSAGES.labels(name).inc(getattr(result, name))
    for ms in result.latencies_ms:
        obs.STREAM_E2E_LATENCY.observe(ms / 1000)
    obs.STREAM_BATCH_SECONDS.observe(seconds)
    obs.STREAM_LAST_PROCESSED.set(time.time())
    if watermark:
        obs.STREAM_EVENT_WATERMARK.set(
            datetime.fromisoformat(watermark).replace(tzinfo=UTC).timestamp()
        )


def run(
    bootstrap: str,
    db_path: Path | str,
    topic: str = TOPIC_TRANSACTIONS,
    dlq_topic: str = TOPIC_DLQ,
    batch_size: int = 500,
    idle_exit: float | None = None,
    max_messages: int | None = None,
    group_id: str = GROUP_ID,
    poll_timeout: float = 0.2,
) -> dict:
    """Consume until idle for ``idle_exit`` seconds, ``max_messages`` are read, or Ctrl-C.

    Stopping on idle means the replay window has ended, so pending reversal checks are
    resolved (``finalize``). Ctrl-C and ``max_messages`` leave them pending for the next run.
    """
    processor = StreamProcessor(db_path, dead_letter_topic=dlq_topic)
    consumer = make_consumer(bootstrap, group_id)
    totals = {"applied": 0, "duplicates": 0, "dead_letters": 0, "reversals_matched": 0}
    seen = 0
    try:
        # On a fresh broker the consumer may start before any producer: create the topics
        # (idempotent, same settings as the producer) rather than failing on unknown topics.
        ensure_topics(bootstrap, [topic, dlq_topic])
        assign_from_store(consumer, processor, [topic, dlq_topic])
        last_message = time.monotonic()
        while True:
            # Returns as soon as batch_size messages are ready, or after poll_timeout with fewer:
            # this bounds the extra latency a trickle of events sees.
            msgs = consumer.consume(num_messages=batch_size, timeout=poll_timeout)
            batch = []
            for m in msgs:
                if m.error():
                    if m.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(m.error())
                batch.append(Message(m.topic(), m.partition(), m.offset(), m.value()))
            if batch:
                started = time.perf_counter()
                result = processor.process(batch)
                record_metrics(result, time.perf_counter() - started, processor.event_watermark())
                for k in totals:
                    totals[k] += getattr(result, k)
                consumer.commit(
                    offsets=[
                        TopicPartition(t, p, o) for (t, p), o in processor.stored_offsets().items()
                    ],
                    asynchronous=True,
                )
                seen += len(batch)
                last_message = time.monotonic()
                if seen // 50_000 != (seen - len(batch)) // 50_000:
                    logger.info("Processed %d messages", seen)
            if max_messages is not None and seen >= max_messages:
                break
            if idle_exit is not None and time.monotonic() - last_message >= idle_exit:
                totals["reversals_matched"] += processor.finalize()
                break
    finally:
        consumer.close()
    status = processor.status() | {"this_run": totals}
    processor.close()
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consume replayed transactions into daily demand")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--topic", default=TOPIC_TRANSACTIONS)
    parser.add_argument("--dlq-topic", default=TOPIC_DLQ)
    parser.add_argument("--batch-size", type=int, default=500, help="messages per DB transaction")
    parser.add_argument("--idle-exit", type=float, help="stop after this many idle seconds")
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--poll-timeout", type=float, default=0.2, help="max wait per poll (s)")
    parser.add_argument("--metrics-port", type=int, help="serve Prometheus metrics on this port")
    parser.add_argument("--status", action="store_true", help="print state and exit")
    args = parser.parse_args(argv)
    obs.configure_logging()
    if args.metrics_port:
        start_http_server(args.metrics_port)

    if args.status:
        with StreamProcessor(args.db) as processor:
            print(json.dumps(processor.status(), indent=2))
        return 0
    print("SIMULATED REAL-TIME: consuming a replay of historical UCI transactions, not live data.")
    try:
        status = run(
            args.bootstrap,
            args.db,
            args.topic,
            args.dlq_topic,
            args.batch_size,
            args.idle_exit,
            args.max_messages,
            poll_timeout=args.poll_timeout,
        )
    except KeyboardInterrupt:
        print("Interrupted. The last committed poll batch is kept; rerun to continue.")
        return 130
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
