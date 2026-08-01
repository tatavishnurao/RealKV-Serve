"""Tests for KV-cache stress experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Import helpers from the stress runner without executing run().
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_kv_stress as stress  # noqa: E402


REPORT_DIR = Path("reports/kv_stress")


def _experiment_json_files() -> list[Path]:
    """Individual experiment report files (exclude summary_*)."""
    if not REPORT_DIR.is_dir():
        return []
    return sorted(
        p
        for p in REPORT_DIR.glob("*.json")
        if p.is_file() and not p.name.startswith("summary_")
    )


def test_script_helpers_run_without_crash() -> None:
    """Core KV manipulation helpers execute on fake tensors without error."""
    k = torch.randn(1, 4, 16, 8)
    v = torch.randn(1, 4, 16, 8)
    past = ((k, v), (k.clone(), v.clone()))

    truncated = stress.truncate_kv(past, 8)
    layers = stress.to_legacy_kv(truncated)
    assert layers[0][0].shape[2] == 8
    assert layers[0][1].shape[2] == 8

    wiped = stress.zero_kv(past)
    wk, wv = stress.to_legacy_kv(wiped)[0]
    assert torch.count_nonzero(wk) == 0
    assert torch.count_nonzero(wv) == 0

    noisy = stress.inject_noise_kv(past, 1e-2)
    nk, _nv = stress.to_legacy_kv(noisy)[0]
    assert not torch.allclose(nk, k)

    assert stress.kv_cache_bytes(past) > 0
    assert stress.describe_divergence("abc", "abc") == "identical"
    assert "differs" in stress.describe_divergence("abc", "axc")


def test_script_module_loads() -> None:
    """Stress runner module imports cleanly (no crash on import)."""
    assert hasattr(stress, "run")
    assert hasattr(stress, "greedy_generate")
    assert stress.MAX_NEW_TOKENS == 32


def test_json_output_exists() -> None:
    """At least one experiment JSON report exists after a stress run."""
    reports = _experiment_json_files()
    assert len(reports) >= 1, "No JSON reports found in reports/kv_stress/"


def test_at_least_three_experiments_executed() -> None:
    """At least three distinct experiment runs produced JSON output."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    experiments: list[tuple[str, object]] = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "experiment" in data:
            experiments.append((data["experiment"], data.get("param")))

    # Prefer summary if present: count results inside latest summary.
    summaries = sorted(REPORT_DIR.glob("summary_*.json"))
    if summaries:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        results = summary.get("results", [])
        assert len(results) >= 3, f"Expected >=3 experiments in summary, got {len(results)}"
    else:
        assert len(experiments) >= 3, f"Expected >=3 experiment files, got {len(experiments)}"


def test_outputs_not_identical_across_experiments() -> None:
    """Different KV manipulations must not all produce the same text."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    outputs: list[str] = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "output" in data and "experiment" in data:
            outputs.append(data["output"])

    summaries = sorted(REPORT_DIR.glob("summary_*.json"))
    if summaries:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        for r in summary.get("results", []):
            if "output" in r:
                outputs.append(r["output"])

    assert len(outputs) >= 2, "Need at least two experiment outputs to compare"
    unique = set(outputs)
    assert len(unique) > 1, (
        "All experiment outputs were identical; expected visible divergence "
        f"from KV manipulations. outputs={outputs!r}"
    )


def test_report_schema_fields() -> None:
    """Each experiment JSON contains the required metric fields."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    required = {
        "experiment",
        "param",
        "output",
        "tokens_generated",
        "latency_ms",
        "kv_bytes",
        "divergence",
    }
    sample = json.loads(reports[-1].read_text(encoding="utf-8"))
    missing = required - set(sample.keys())
    assert not missing, f"Missing fields in {reports[-1].name}: {missing}"
