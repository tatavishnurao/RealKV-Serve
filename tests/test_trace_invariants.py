from realkv.trace_schema import TraceEvent


def test_block_ids_reject_negative():
    try:
        TraceEvent(
            "t", "r", 0, 1, "context", "block_state_observed", request_block_ids=[-1]
        ).to_dict()
    except ValueError:
        return
    assert False


def test_event_types_are_restricted():
    try:
        TraceEvent("t", "r", 0, 1, "context", "invented").to_dict()
    except ValueError:
        return
    assert False
