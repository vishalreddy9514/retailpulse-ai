"""Kafka producer that replays historical UCI Online Retail II transactions.

SIMULATED REAL-TIME STREAMING USING HISTORICAL DATA. Transactions from 2009-2011 are
replayed in timestamp order. The application does not receive live transactions from
the retailer.

Usage::

    docker compose up -d kafka
    # fast replay (development): as fast as possible, one day of history
    python -m src.streaming.kafka_producer --mode fast --start 2010-12-01 --end 2010-12-01
    # demo: one hour of history per second, gaps capped at 2 s
    python -m src.streaming.kafka_producer --mode accelerated --speed 3600 --max-gap 2
    # no broker: validate and count only
    python -m src.streaming.kafka_producer --dry-run

Restarting the same command resumes after the last committed event. Pass ``--from-start``
to replay again on purpose (the consumer de-duplicates on ``event_id``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

from src.ingestion.uci_loader import load_raw
from src.streaming.replay import InMemorySink, ReplayConfig, prepare, replay, window
from src.streaming.schema import TOPIC_DLQ, TOPIC_TRANSACTIONS

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP = os.environ.get("RETAILPULSE_KAFKA_BOOTSTRAP", "localhost:9092")
PARTITIONS = 6
BANNER = (
    "SIMULATED REAL-TIME: replaying historical UCI Online Retail II transactions "
    "(2009-2011) in timestamp order. This is not live retailer data."
)


def ensure_topics(bootstrap: str, topics: list[str], partitions: int = PARTITIONS) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    missing = [
        NewTopic(t, num_partitions=partitions, replication_factor=1)
        for t in topics
        if t not in existing
    ]
    for topic, fut in admin.create_topics(missing).items() if missing else []:
        try:
            fut.result()
            logger.info("Created topic %s", topic)
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


class KafkaSink:
    """Transactional Kafka sink.

    ``init_transactions`` fences any earlier producer with the same transactional id and
    aborts its open transaction. Only then is the checkpoint read, so a crashed run's
    half-written batch can never be mistaken for committed data.
    """

    def __init__(
        self, bootstrap: str, topic: str, dlq_topic: str, transactional_id: str | None = None
    ):
        self.bootstrap = bootstrap
        self.topics = [topic, dlq_topic]
        self.producer = Producer(
            {
                "bootstrap.servers": bootstrap,
                "transactional.id": transactional_id or f"retailpulse-replay-{topic}",
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": 20,
                "compression.type": "lz4",
                "queue.buffering.max.messages": 500_000,
                "transaction.timeout.ms": 60_000,
            }
        )
        self.producer.init_transactions(30)

    def last_committed_sequence(self) -> int | None:
        """Highest ``sequence`` header among committed messages on the event and DLQ topics."""
        consumer = Consumer(
            {
                "bootstrap.servers": self.bootstrap,
                "group.id": f"retailpulse-checkpoint-{uuid.uuid4().hex[:8]}",
                "isolation.level": "read_committed",
                "enable.auto.commit": False,
                "enable.partition.eof": True,
            }
        )
        best: int | None = None
        try:
            for topic in self.topics:
                parts = consumer.list_topics(topic, timeout=10).topics[topic].partitions
                for p in parts:
                    found = self._last_sequence_in_partition(consumer, topic, p)
                    if found is not None:
                        best = found if best is None else max(best, found)
        finally:
            consumer.close()
        return best

    @staticmethod
    def _last_sequence_in_partition(consumer: Consumer, topic: str, partition: int) -> int | None:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=10)
        back = 64
        while high > low:
            start = max(low, high - back)
            consumer.assign([TopicPartition(topic, partition, start)])
            best = None
            while True:
                msg = consumer.poll(10)
                if msg is None:
                    raise TimeoutError(f"timed out reading {topic}[{partition}] for checkpoint")
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        break
                    raise KafkaException(msg.error())
                seq = dict(msg.headers() or []).get("sequence")
                if seq is not None:
                    best = max(best or 0, int(seq))
            if best is not None or start == low:
                return best
            back *= 4  # only aborted or control records in the window; look further back
        return None

    def begin(self) -> None:
        self.producer.begin_transaction()

    def send(self, topic, key, value, headers) -> None:
        while True:
            try:
                self.producer.produce(topic, value=value, key=key, headers=headers)
                break
            except BufferError:
                self.producer.poll(0.5)  # local queue full: let deliveries drain
        self.producer.poll(0)

    def commit(self) -> None:
        self.producer.commit_transaction(60)

    def abort(self) -> None:
        try:
            self.producer.abort_transaction(30)
        except KafkaException as exc:  # e.g. no transaction in progress
            logger.warning("abort_transaction: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay historical transactions to Kafka")
    parser.add_argument("--mode", choices=["fast", "accelerated"], default="fast")
    parser.add_argument(
        "--speed", type=float, default=3600.0, help="accelerated: history seconds per wall second"
    )
    parser.add_argument(
        "--max-gap", type=float, default=2.0, help="accelerated: cap on any wait (s)"
    )
    parser.add_argument("--rate", type=float, help="fast: max events per second")
    parser.add_argument("--batch-size", type=int, default=1000, help="events per transaction")
    parser.add_argument("--start", help="first event date (inclusive, YYYY-MM-DD)")
    parser.add_argument("--end", help="last event date (inclusive, YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="maximum events to send")
    parser.add_argument("--from-start", action="store_true", help="ignore the committed checkpoint")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--topic", default=TOPIC_TRANSACTIONS)
    parser.add_argument("--dlq-topic", default=TOPIC_DLQ)
    parser.add_argument("--dry-run", action="store_true", help="validate and count, no broker")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(BANNER)

    config = ReplayConfig(
        mode=args.mode,
        speed=args.speed,
        max_gap_seconds=args.max_gap,
        max_events_per_second=args.rate,
        batch_size=args.batch_size,
        from_start=args.from_start,
        topic=args.topic,
        dlq_topic=args.dlq_topic,
    )
    frame = window(prepare(load_raw()), args.start, args.end, args.limit)
    logger.info("Replay window: %d events", len(frame))

    if args.dry_run:
        sink = InMemorySink(keep_payloads=False)
    else:
        ensure_topics(args.bootstrap, [args.topic, args.dlq_topic])
        sink = KafkaSink(args.bootstrap, args.topic, args.dlq_topic)

    try:
        stats = replay(frame, sink, config)
    except KeyboardInterrupt:
        print(
            f"Interrupted. The open batch was aborted. Committed up to sequence "
            f"{sink.last_committed_sequence()}; rerun the same command to resume."
        )
        return 130
    out = asdict(stats) | {"events_per_second": round(stats.events_per_second, 1)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
