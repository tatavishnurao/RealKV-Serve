"""Narrow opt-in observation helpers for verified TRT-LLM objects.

The wrapper is deliberately generic: the target container must pass exact methods
from its pinned source map. Unknown values remain null and return values are untouched.
"""

from __future__ import annotations
import time
import uuid
from .trace_schema import TraceEvent, dumps


class JsonlObserver:
    def __init__(self, path: str, request_id: str | None = None, **context):
        self.path, self.request_id, self.context = (
            path,
            request_id or f"req-{uuid.uuid4().hex[:12]}",
            context,
        )
        self.trace_id, self.index = uuid.uuid4().hex, 0
        self.handle = open(path, "a", encoding="utf-8", buffering=1)

    def emit(
        self, event_type: str, phase: str, *, source_component=None, source_method=None, **values
    ):
        event_metadata = dict(self.context)
        event_metadata.update(values.pop("metadata", {}) or {})
        event = TraceEvent(
            trace_id=self.trace_id,
            request_id=self.request_id,
            event_index=self.index,
            timestamp_ns=time.monotonic_ns(),
            phase=phase,
            event_type=event_type,
            source_component=source_component,
            source_method=source_method,
            metadata=event_metadata,
            **values,
        )
        self.handle.write(dumps(event) + "\n")
        self.index += 1

    def close(self):
        self.handle.flush()
        self.handle.close()


def wrap_method(
    obj, method_name: str, observer: JsonlObserver, event_type: str, phase: str, component: str
):
    """Wrap one source-verified method while preserving its result and exceptions."""
    original = getattr(obj, method_name)

    def wrapped(*args, **kwargs):
        observer.emit(event_type, phase, source_component=component, source_method=method_name)
        result = original(*args, **kwargs)
        observer.emit(
            event_type,
            phase,
            source_component=component,
            source_method=method_name,
            metadata={"return_type": type(result).__name__},
        )
        return result

    setattr(obj, method_name, wrapped)
    return original
