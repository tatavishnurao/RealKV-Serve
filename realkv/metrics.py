"""Host-observed timing metrics; these are not kernel timings."""

from __future__ import annotations
import statistics


def summarize_timestamps(start_ns: int, token_ns: list[int], end_ns: int) -> dict:
    itl = [(b - a) / 1e6 for a, b in zip(token_ns, token_ns[1:])]
    ttft = (token_ns[0] - start_ns) / 1e6 if token_ns else None
    return {
        "timing_label": "HOST_OBSERVED_END_TO_END_TIME",
        "request_start_ns": start_ns,
        "first_token_timestamp_ns": token_ns[0] if token_ns else None,
        "token_timestamps_ns": token_ns,
        "request_end_ns": end_ns,
        "ttft_ms": ttft,
        "inter_token_latency_ms": itl,
        "mean_itl_ms": statistics.mean(itl) if itl else None,
        "median_itl_ms": statistics.median(itl) if itl else None,
        "p95_itl_ms": percentile(itl, 95) if itl else None,
        "output_tokens_per_second_after_first_token": (
            (len(token_ns) - 1) / ((token_ns[-1] - token_ns[0]) / 1e9)
        )
        if len(token_ns) > 1 and token_ns[-1] > token_ns[0]
        else None,
    }


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lower, upper = int(index), min(int(index) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)
