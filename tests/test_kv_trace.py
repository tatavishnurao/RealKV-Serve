"""Tests for KV-cache trace lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_output_file_exists() -> None:
    """At least one JSONL trace file exists after a run."""
    reports = sorted(Path("reports/kv_trace").glob("run_*.jsonl"))
    assert len(reports) >= 1, "No trace files found in reports/kv_trace/"


def test_at_least_ten_steps() -> None:
    """Trace contains at least 10 decode steps."""
    reports = sorted(Path("reports/kv_trace").glob("run_*.jsonl"))
    if not reports:
        pytest.skip("No trace files exist")
    latest = reports[-1]
    lines = [line for line in latest.read_text().splitlines() if line.strip()]
    assert len(lines) >= 10, f"Expected >=10 steps, got {len(lines)}"


def test_kv_size_monotonic_increase() -> None:
    """total_kv_bytes increases monotonically across decode steps."""
    reports = sorted(Path("reports/kv_trace").glob("run_*.jsonl"))
    if not reports:
        pytest.skip("No trace files exist")
    latest = reports[-1]
    records = [json.loads(line) for line in latest.read_text().splitlines() if line.strip()]
    sizes = [r["total_kv_bytes"] for r in records]
    for i in range(1, len(sizes)):
        assert sizes[i] > sizes[i - 1], (
            f"KV size did not increase at step {i}: "
            f"{sizes[i-1]} -> {sizes[i]}"
        )
