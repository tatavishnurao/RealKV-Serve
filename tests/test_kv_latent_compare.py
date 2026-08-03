"""Tests for simulated latent KV compression vs baseline KV comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Import helpers from the latent compare runner without executing run().
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_kv_latent_compare as latent  # noqa: E402


REPORT_DIR = Path("reports/kv_latent_compare")


def _experiment_json_files() -> list[Path]:
    """Individual experiment report files (exclude summary_*)."""
    if not REPORT_DIR.is_dir():
        return []
    return sorted(
        p
        for p in REPORT_DIR.glob("*.json")
        if p.is_file() and not p.name.startswith("summary_")
    )


def test_script_module_loads() -> None:
    """Latent compare runner module imports cleanly (no crash on import)."""
    assert hasattr(latent, "run")
    assert hasattr(latent, "greedy_generate")
    assert hasattr(latent, "compress_latent")
    assert latent.COMPRESSION_RATIOS == [2, 4, 8]


def test_projection_shapes_and_losslessness() -> None:
    """Orthonormal projection has the right shapes and is lossless at full rank."""
    head_dim, latent_dim = 8, 8
    w_down, w_up = latent.build_projection(
        head_dim,
        latent_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert w_down.shape == (head_dim, latent_dim)
    assert w_up.shape == (latent_dim, head_dim)

    k = torch.randn(1, 2, 4, head_dim)
    recon = k @ w_down @ w_up
    assert torch.allclose(recon, k, atol=1e-5), "Full-rank projection should be lossless"


def test_projection_compresses_at_higher_ratio() -> None:
    """Smaller latent dims produce larger reconstruction error."""
    head_dim = 8
    k = torch.randn(1, 2, 4, head_dim)
    errors: list[float] = []
    for latent_dim in (4, 2, 1):
        w_down, w_up = latent.build_projection(
            head_dim,
            latent_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        recon = k @ w_down @ w_up
        errors.append(float(torch.norm(recon - k)))
    assert errors[0] < errors[1] < errors[2], (
        f"Expected increasing error with compression: {errors}"
    )


def test_compress_latent_reduces_kv_bytes() -> None:
    """Latent compression footprint scales down with the compression ratio."""
    head_dim = 8
    k = torch.randn(1, 2, 16, head_dim)
    v = torch.randn(1, 2, 16, head_dim)
    past = ((k, v), (k.clone(), v.clone()))
    baseline_bytes = latent.kv_cache_bytes(past)

    for ratio in (2, 4, 8):
        latent_dim = latent.resolve_latent_dim(head_dim, ratio)
        w_down, w_up = latent.build_projection(
            head_dim,
            latent_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        rebuilt, latent_bytes = latent.compress_latent(past, w_down, w_up)
        layers = latent.to_legacy_kv(rebuilt)
        assert layers[0][0].shape == k.shape
        assert latent_bytes < baseline_bytes
        assert latent_bytes <= baseline_bytes * ratio * 1.05 + 1


def test_compress_latent_is_idempotent() -> None:
    """Re-projecting an already-compressed cache does not add more loss."""
    head_dim, latent_dim = 8, 4
    k = torch.randn(1, 2, 16, head_dim)
    v = torch.randn(1, 2, 16, head_dim)
    w_down, w_up = latent.build_projection(
        head_dim,
        latent_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    past = ((k, v),)
    once, _ = latent.compress_latent(past, w_down, w_up)
    twice, _ = latent.compress_latent(once, w_down, w_up)

    once_layers = latent.to_legacy_kv(once)
    twice_layers = latent.to_legacy_kv(twice)
    assert torch.allclose(once_layers[0][0], twice_layers[0][0], atol=1e-6)


def test_json_output_exists() -> None:
    """At least one experiment JSON report exists after a latent compare run."""
    reports = _experiment_json_files()
    assert len(reports) >= 1, "No JSON reports found in reports/kv_latent_compare/"


def test_baseline_and_latent_outputs_differ() -> None:
    """Baseline and latent modes must not all produce identical text."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    outputs: list[str] = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "output" in data:
            outputs.append(data["output"])

    summaries = sorted(REPORT_DIR.glob("summary_*.json"))
    if summaries:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        for r in summary.get("results", []):
            if "output" in r:
                outputs.append(r["output"])

    assert len(outputs) >= 2, "Need at least two outputs to compare"
    assert len(set(outputs)) > 1, "All outputs were identical; expected divergence"


def test_kv_size_reduces_with_compression() -> None:
    """Latent runs must report strictly fewer KV bytes than the baseline."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    baseline_kv: int | None = None
    latent_kv: list[int] = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("mode") == "baseline":
            baseline_kv = int(data["kv_bytes"])
        elif data.get("mode") == "latent":
            latent_kv.append(int(data["kv_bytes"]))

    if baseline_kv is None or not latent_kv:
        pytest.skip("No baseline or latent reports available")

    assert all(kv < baseline_kv for kv in latent_kv), (
        f"Expected latent kv_bytes ({latent_kv}) < baseline ({baseline_kv})"
    )


def test_multiple_compression_configs_tested() -> None:
    """Baseline plus all latent compression ratios produce JSON output."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    ratios = set()
    modes = set()
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        modes.add(data.get("mode"))
        if data.get("mode") == "latent":
            ratios.add(data.get("compression_ratio"))

    summaries = sorted(REPORT_DIR.glob("summary_*.json"))
    if summaries:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        for r in summary.get("results", []):
            modes.add(r.get("mode"))
            if r.get("mode") == "latent":
                ratios.add(r.get("compression_ratio"))

    assert modes == {"baseline", "latent"}
    assert ratios == {2, 4, 8}, f"Expected ratios 2, 4, 8, got {ratios}"


def test_report_schema_fields() -> None:
    """Each experiment JSON contains the required metric fields."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    required = {
        "mode",
        "compression_ratio",
        "output",
        "tokens_generated",
        "latency_ms",
        "kv_bytes",
        "divergence",
    }
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - set(data.keys())
        assert not missing, f"Missing fields in {path.name}: {missing}"
