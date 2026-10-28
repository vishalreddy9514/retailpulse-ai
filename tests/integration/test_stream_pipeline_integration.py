"""Producer -> Kafka -> consumer -> reconcile against batch, on a real broker.

Skipped unless RETAILPULSE_KAFKA_BOOTSTRAP is set (see test_kafka_producer_integration.py).
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("RETAILPULSE_KAFKA_BOOTSTRAP")
if not BOOTSTRAP:
    pytest.skip("RETAILPULSE_KAFKA_BOOTSTRAP not set", allow_module_level=True)

from src.streaming import kafka_consumer  # noqa: E402
from src.streaming.kafka_producer import KafkaSink, ensure_topics  # noqa: E402
from src.streaming.reconcile import load_stream, reconcile  # noqa: E402
from src.streaming.replay import ReplayConfig, prepare, replay  # noqa: E402
from tests.unit.test_stream_processing import ROWS, batch_daily  # noqa: E402


def test_end_to_end_stream_matches_batch_and_survives_restart(make_raw, tmp_path):
    run = uuid.uuid4().hex[:8]
    topic, dlq = f"it.e2e.{run}", f"it.e2e.dlq.{run}"
    ensure_topics(BOOTSTRAP, [topic, dlq])
    raw = make_raw(ROWS)
    config = ReplayConfig(batch_size=5, topic=topic, dlq_topic=dlq)
    replay(prepare(raw), KafkaSink(BOOTSTRAP, topic, dlq, transactional_id=f"it-{run}"), config)

    db = tmp_path / "state.sqlite"
    group = f"it-{run}"
    # First run stops early (as if killed); the second resumes from the stored offsets.
    kafka_consumer.run(BOOTSTRAP, db, topic, dlq, batch_size=4, max_messages=6, group_id=group)
    status = kafka_consumer.run(
        BOOTSTRAP, db, topic, dlq, batch_size=4, idle_exit=3, group_id=group
    )

    assert status["counters"]["duplicates_ignored"] == 1  # the fixture's one exact duplicate
    assert status["pending_reversal_checks"] == 0
    report = reconcile(load_stream(db), batch_daily(raw))
    assert report["exact_match"], report

    # Replaying everything again on purpose changes nothing downstream.
    replay(
        prepare(raw),
        KafkaSink(BOOTSTRAP, topic, dlq, transactional_id=f"it-{run}"),
        ReplayConfig(batch_size=5, topic=topic, dlq_topic=dlq, from_start=True),
    )
    kafka_consumer.run(BOOTSTRAP, db, topic, dlq, idle_exit=3, group_id=group)
    assert reconcile(load_stream(db), batch_daily(raw))["exact_match"]
