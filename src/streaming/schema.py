"""Event contract for the simulated real-time transaction stream.

SIMULATED REAL-TIME: every event is a *historical* UCI Online Retail II line (2009-2011)
replayed in timestamp order. Nothing here is live retailer data, and every event says so
in ``data_origin``.

Events carry the raw line as recorded (cancellations, adjustments and all). Classifying
lines as sales, returns and so on is the consumer's job, using the same rules as the batch
pipeline. A record that fails this schema is routed to the dead-letter topic with its
validation errors, and is never dropped silently.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA_VERSION = 1
DATA_ORIGIN = "historical_replay"
TOPIC_TRANSACTIONS = "retail.transactions.v1"
TOPIC_DLQ = "retail.transactions.dlq.v1"


class TransactionEvent(BaseModel):
    """One replayed transaction line."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    data_origin: Literal["historical_replay"] = DATA_ORIGIN
    event_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="deterministic line id")
    sequence: int = Field(ge=0, description="position in the canonical replay order")
    replay_run_id: str = Field(min_length=1)

    invoice: str = Field(pattern=r"^[A-Z]?\d{6}$")
    stock_code: str = Field(min_length=1, max_length=20)
    description: str | None = None
    quantity: int
    unit_price: float
    customer_id: str | None = Field(default=None, pattern=r"^\d+$")
    country: str = Field(min_length=1)

    event_time: datetime = Field(
        description="original invoice timestamp (UK local time as recorded, no timezone in source)"
    )
    ingestion_time: datetime = Field(description="UTC time the event was published to Kafka")
    source_sheet: str
    source_row: int = Field(ge=2)

    @field_validator("unit_price")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("unit_price must be finite")
        return v

    @field_validator("event_time")
    @classmethod
    def _naive_local(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("event_time is source local time and must not carry a timezone")
        return v

    @field_validator("ingestion_time")
    @classmethod
    def _aware_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset().total_seconds() != 0:
            raise ValueError("ingestion_time must be timezone-aware UTC")
        return v

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> TransactionEvent:
        return cls.model_validate_json(payload)


def dead_letter(
    raw: dict[str, Any], error: ValidationError | str, sequence: int, run_id: str, at: datetime
) -> bytes:
    """Dead-letter payload: the raw record as text, why it was rejected, and when."""
    if isinstance(error, ValidationError):
        errors = [
            {"field": ".".join(str(p) for p in e["loc"]), "error": e["msg"]} for e in error.errors()
        ]
    else:
        errors = [{"field": "", "error": str(error)}]
    body = {
        "schema_version": SCHEMA_VERSION,
        "data_origin": DATA_ORIGIN,
        "sequence": sequence,
        "replay_run_id": run_id,
        "rejected_at": at.isoformat(),
        "errors": errors,
        "raw": {k: (None if _is_missing(v) else str(v)) for k, v in raw.items()},
    }
    return json.dumps(body).encode("utf-8")


def _is_missing(v: Any) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v)) or str(v) in ("NaT", "<NA>")
    except (TypeError, ValueError):
        return False


def json_schema() -> dict:
    return TransactionEvent.model_json_schema()
