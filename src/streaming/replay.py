"""Broker-independent engine that replays historical transactions as a stream.

SIMULATED REAL-TIME. This replays historical UCI Online Retail II lines in timestamp order.
It does not receive live transactions from the retailer.

Guarantees, with a transactional sink such as ``KafkaSink``:

* **Order.** Events are sent in canonical order (event time, then source sheet and row).
  ``sequence`` is the line's position in that order across the whole dataset, so it means
  the same thing in every run, whatever window is replayed.
* **Atomic batches.** Events (and any dead-letter records) are committed in transactions
  of ``batch_size``. A crash aborts the open batch, and none of it becomes visible.
* **Restart without duplicates.** On start, the engine asks the sink for the highest
  committed ``sequence`` and resumes after it. Batches commit in order, so everything up
  to that sequence is already committed. ``from_start`` deliberately replays again. The
  consumer de-duplicates on the deterministic ``event_id`` either way.
* **Pacing.** ``accelerated`` maps event-time gaps to wall time divided by ``speed``, with
  each gap capped at ``max_gap_seconds`` so nights and weekends don't stall a demo.
  ``fast`` sends as fast as possible, optionally capped at ``max_events_per_second``.
  An open batch is committed before any noticeable wait, and at least every
  ``max_batch_age_seconds``, so consumers see events promptly (read_committed consumers
  cannot see an event until its batch commits).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

import pandas as pd
from pydantic import ValidationError

from src.processing.clean_transactions import make_event_id, normalise
from src.streaming.schema import (
    DATA_ORIGIN,
    SCHEMA_VERSION,
    TOPIC_DLQ,
    TOPIC_TRANSACTIONS,
    TransactionEvent,
    dead_letter,
)

logger = logging.getLogger(__name__)

EVENT_FIELDS = [
    "sequence",
    "event_id",
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "unit_price",
    "customer_id",
    "country",
    "invoice_date",
    "source_sheet",
    "source_row",
]


class Sink(Protocol):
    def last_committed_sequence(self) -> int | None: ...
    def begin(self) -> None: ...
    def send(
        self, topic: str, key: bytes, value: bytes, headers: list[tuple[str, bytes]]
    ) -> None: ...
    def commit(self) -> None: ...
    def abort(self) -> None: ...


class Clock:
    """Wall clock. Tests substitute a fake one."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def utcnow(self) -> datetime:
        return datetime.now(UTC)


@dataclass
class ReplayConfig:
    mode: Literal["fast", "accelerated"] = "fast"
    speed: float = 3600.0  # accelerated: seconds of history per second of wall time
    max_gap_seconds: float = 2.0  # accelerated: cap on any single wait
    max_events_per_second: float | None = None  # fast: optional rate cap
    batch_size: int = 1000
    commit_before_wait_seconds: float = 0.25
    max_batch_age_seconds: float = 0.5  # commit an open batch at least this often
    from_start: bool = False
    topic: str = TOPIC_TRANSACTIONS
    dlq_topic: str = TOPIC_DLQ

    def __post_init__(self) -> None:
        if self.mode not in ("fast", "accelerated"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.speed <= 0 or self.max_gap_seconds < 0 or self.batch_size < 1:
            raise ValueError("speed must be > 0, max_gap_seconds >= 0, batch_size >= 1")
        if self.max_events_per_second is not None and self.max_events_per_second <= 0:
            raise ValueError("max_events_per_second must be > 0")


@dataclass
class ReplayStats:
    run_id: str
    resumed_after_sequence: int | None = None
    published: int = 0
    rejected: int = 0
    batches: int = 0
    waited_seconds: float = 0.0
    wall_seconds: float = 0.0
    first_sequence: int | None = None
    last_sequence: int | None = None
    first_event_time: str | None = None
    last_event_time: str | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def events_per_second(self) -> float:
        total = self.published + self.rejected
        return total / self.wall_seconds if self.wall_seconds > 0 else 0.0


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical replay frame: normalised fields, deterministic ids, global sequence.

    Normalisation and ``event_id`` are the batch pipeline's own functions, so an event and
    its cleaned batch row share the same id.
    """
    df = normalise(raw)
    df = df.sort_values(
        ["invoice_date", "source_sheet", "source_row"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    df["event_id"] = make_event_id(df)
    df["sequence"] = range(len(df))
    return df[EVENT_FIELDS]


def window(
    frame: pd.DataFrame, start: str | None = None, end: str | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Select a replay window by event date (inclusive) and/or a maximum number of events."""
    out = frame
    if start:
        out = out[out["invoice_date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["invoice_date"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if limit is not None:
        out = out.head(limit)
    return out


def _records(frame: pd.DataFrame) -> Iterator[dict]:
    for values in frame.itertuples(index=False, name=None):
        yield dict(zip(EVENT_FIELDS, values, strict=True))


def _none(v):
    return (
        None if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v) else v
    )


def build_event(rec: dict, run_id: str, ingestion_time: datetime) -> TransactionEvent:
    ts = _none(rec["invoice_date"])
    return TransactionEvent(
        event_id=rec["event_id"],
        sequence=int(rec["sequence"]),
        replay_run_id=run_id,
        invoice=_none(rec["invoice"]),
        stock_code=_none(rec["stock_code"]),
        description=_none(rec["description"]),
        quantity=_none(rec["quantity"]),
        unit_price=_none(rec["unit_price"]),
        customer_id=_none(rec["customer_id"]),
        country=_none(rec["country"]),
        event_time=ts.to_pydatetime() if ts is not None else None,
        ingestion_time=ingestion_time,
        source_sheet=_none(rec["source_sheet"]),
        source_row=int(rec["source_row"]),
    )


def replay(
    frame: pd.DataFrame,
    sink: Sink,
    config: ReplayConfig | None = None,
    clock: Clock | None = None,
    run_id: str | None = None,
    progress_every: int = 100_000,
) -> ReplayStats:
    """Publish ``frame`` (from ``prepare``/``window``) to ``sink``. Returns run statistics."""
    config = config or ReplayConfig()
    clock = clock or Clock()
    stats = ReplayStats(run_id=run_id or uuid.uuid4().hex[:12])

    if not config.from_start:
        stats.resumed_after_sequence = sink.last_committed_sequence()
        if stats.resumed_after_sequence is not None:
            frame = frame[frame["sequence"] > stats.resumed_after_sequence]
            logger.info("Resuming after committed sequence %d", stats.resumed_after_sequence)

    started = clock.monotonic()
    virtual = 0.0
    prev_time: pd.Timestamp | None = None
    in_batch = 0
    batch_started = started
    sink.begin()
    try:
        for rec in _records(frame):
            ts = _none(rec["invoice_date"])
            if config.mode == "accelerated" and ts is not None:
                if prev_time is not None:
                    gap = max((ts - prev_time).total_seconds(), 0.0)
                    virtual += min(gap / config.speed, config.max_gap_seconds)
                prev_time = ts
            elif config.mode == "fast" and config.max_events_per_second:
                virtual += 1.0 / config.max_events_per_second

            wait = started + virtual - clock.monotonic()
            if wait > 0:
                if in_batch and wait >= config.commit_before_wait_seconds:
                    sink.commit()
                    stats.batches += 1
                    in_batch = 0
                    sink.begin()
                clock.sleep(wait)
                stats.waited_seconds += wait
            if in_batch and clock.monotonic() - batch_started >= config.max_batch_age_seconds:
                sink.commit()
                stats.batches += 1
                in_batch = 0
                sink.begin()
            if in_batch == 0:
                batch_started = clock.monotonic()

            now = clock.utcnow()
            seq = int(rec["sequence"])
            headers = [
                ("sequence", str(seq).encode()),
                ("schema_version", str(SCHEMA_VERSION).encode()),
                ("data_origin", DATA_ORIGIN.encode()),
            ]
            try:
                event = build_event(rec, stats.run_id, now)
            except ValidationError as err:
                sink.send(
                    config.dlq_topic,
                    str(seq).encode(),
                    dead_letter(rec, err, seq, stats.run_id, now),
                    headers,
                )
                stats.rejected += 1
                for e in err.errors():
                    reason = f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}"
                    stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1
            else:
                headers.append(("event_id", event.event_id.encode()))
                sink.send(config.topic, event.stock_code.encode(), event.to_bytes(), headers)
                stats.published += 1
                if stats.first_event_time is None:
                    stats.first_event_time = str(event.event_time)
                stats.last_event_time = str(event.event_time)

            stats.first_sequence = seq if stats.first_sequence is None else stats.first_sequence
            stats.last_sequence = seq
            in_batch += 1
            if in_batch >= config.batch_size:
                sink.commit()
                stats.batches += 1
                in_batch = 0
                sink.begin()
            done = stats.published + stats.rejected
            if progress_every and done % progress_every == 0:
                logger.info("Replayed %d events (last event time %s)", done, stats.last_event_time)
        sink.commit()
        if in_batch:
            stats.batches += 1
    except BaseException:
        # Nothing from the open batch becomes visible; a restart resumes after the last commit.
        sink.abort()
        stats.wall_seconds = clock.monotonic() - started
        raise
    stats.wall_seconds = clock.monotonic() - started
    return stats


class InMemorySink:
    """Transactional sink kept in memory, for tests and ``--dry-run``.

    ``fail_after_sends`` raises on that send, simulating a crash mid-batch.
    """

    def __init__(self, keep_payloads: bool = True, fail_after_sends: int | None = None):
        self.keep_payloads = keep_payloads
        self.fail_after_sends = fail_after_sends
        self.committed: list[tuple[str, bytes, bytes, dict[str, bytes]]] = []
        self.committed_count = 0
        self.max_committed_seq: int | None = None
        self._pending: list = []
        self._sends = 0
        self.commits = 0
        self.aborts = 0

    def last_committed_sequence(self) -> int | None:
        return self.max_committed_seq

    def begin(self) -> None:
        self._pending = []

    def send(self, topic, key, value, headers) -> None:
        self._sends += 1
        if self.fail_after_sends is not None and self._sends > self.fail_after_sends:
            raise RuntimeError("simulated crash")
        self._pending.append((topic, key, value, dict(headers)))

    def commit(self) -> None:
        for msg in self._pending:
            seq = int(msg[3]["sequence"])
            self.max_committed_seq = (
                seq if self.max_committed_seq is None else max(self.max_committed_seq, seq)
            )
        self.committed_count += len(self._pending)
        if self.keep_payloads:
            self.committed.extend(self._pending)
        self._pending = []
        self.commits += 1

    def abort(self) -> None:
        self._pending = []
        self.aborts += 1
