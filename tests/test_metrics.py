from realkv.metrics import percentile, summarize_timestamps


def test_percentile_and_empty_metrics():
    assert percentile([1.0, 2.0, 3.0], 95) == 2.9
    assert summarize_timestamps(10, [], 20)["ttft_ms"] is None
