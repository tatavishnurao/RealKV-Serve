"""Versioned JSONL event schema and validation for observed runtime events."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
EVENT_TYPES = {
    "request_submitted",
    "request_scheduled",
    "resource_prepare_start",
    "resource_prepare_end",
    "block_state_observed",
    "attention_prepare",
    "attention_forward",
    "token_emitted",
    "request_completed",
    "block_state_final",
}
PHASES = {"arrival", "context", "generation", "completion", "unknown"}


@dataclass
class TraceEvent:
    trace_id: str
    request_id: str
    event_index: int
    timestamp_ns: int
    phase: str
    event_type: str
    source_component: str | None = None
    source_method: str | None = None
    step_index: int | None = None
    sequence_length: int | None = None
    context_length: int | None = None
    num_context_tokens: int | None = None
    num_generation_tokens: int | None = None
    tokens_per_block: int | None = None
    request_block_ids: list[int] | None = None
    new_block_ids: list[int] | None = None
    reused_block_ids: list[int] | None = None
    released_block_ids: list[int] | None = None
    num_free_blocks_before: int | None = None
    num_free_blocks_after: int | None = None
    num_total_blocks: int | None = None
    cache_dtype: str | None = None
    cache_pool_shapes: Any = None
    attention_backend: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    runtime_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.event_type not in EVENT_TYPES or self.phase not in PHASES:
            raise ValueError(f"unsupported event type or phase: {self.event_type}/{self.phase}")
        if self.event_index < 0 or self.timestamp_ns < 0:
            raise ValueError("event index and timestamp must be non-negative")
        for name in (
            "request_block_ids",
            "new_block_ids",
            "reused_block_ids",
            "released_block_ids",
        ):
            values = getattr(self, name)
            if values is not None and any(
                not isinstance(value, int) or value < 0 for value in values
            ):
                raise ValueError(f"{name} must contain non-negative integers")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def dumps(event: TraceEvent) -> str:
    return json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))


def load_events(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    for event in events:
        if event.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unexpected schema version")
    return events
