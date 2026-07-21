import json
from realkv.trace_schema import TraceEvent, dumps, load_events


def test_deterministic_serialization(tmp_path):
    event = TraceEvent("t", "r", 0, 1, "arrival", "request_submitted")
    assert dumps(event) == dumps(event)
    path = tmp_path / "trace.jsonl"
    path.write_text(dumps(event) + "\n")
    assert load_events(str(path))[0]["schema_version"] == "1.0"


def test_unknown_fields_are_null():
    data = json.loads(dumps(TraceEvent("t", "r", 0, 1, "arrival", "request_submitted")))
    assert data["request_block_ids"] is None and data["num_free_blocks_before"] is None
