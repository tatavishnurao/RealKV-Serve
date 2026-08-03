"""Tests for structured KV compression comparison (head prune / SVD / pool)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Import helpers from the structured compare runner without executing run().
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_kv_structured_compare as structured  # noqa: E402


REPORT_DIR = Path("reports/kv_structured_compare")

NUM_HEADS = 32
HEAD_DIM = 64


def _experiment_json_files() -> list[Path]:
    """Individual experiment report files (exclude summary_*)."""
    if not REPORT_DIR.is_dir():
        return []
    return sorted(
        p
        for p in REPORT_DIR.glob("*.json")
        if p.is_file() and not p.name.startswith("summary_")
    )


def _make_cache(seq: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(1, 8, seq, HEAD_DIM)
    v = torch.randn(1, 8, seq, HEAD_DIM)
    return k, v


def test_script_module_loads() -> None:
    """Structured compare runner module imports cleanly (no crash on import)."""
    assert hasattr(structured, "run")
    assert hasattr(structured, "greedy_generate")
    assert hasattr(structured, "head_prune_cache")
    assert hasattr(structured, "svd_compress_cache")
    assert hasattr(structured, "pool_tokens_cache")
    assert len(structured.EXPERIMENTS) == 6


def test_head_prune_zeroes_pruned_heads_and_keeps_shape() -> None:
    """Head pruning keeps tensor shapes but zeroes the dropped heads."""
    k, v = _make_cache()
    past = ((k, v),)
    pruned = structured.head_prune_cache(past, 4, 8)
    pk, pv = structured.to_legacy_kv(pruned)[0]
    assert pk.shape == k.shape
    assert torch.count_nonzero(pk[:, 4:, :, :]) == 0
    assert torch.count_nonzero(pv[:, 4:, :, :]) == 0
    assert torch.count_nonzero(pk[:, :4, :, :]) > 0


def test_svd_full_rank_is_lossless() -> None:
    """SVD reconstruction at full rank is close to the original tensor."""
    k, _ = _make_cache()
    rank = HEAD_DIM
    recon = structured.svd_approx(k, rank)
    assert recon.shape == k.shape
    assert torch.allclose(recon, k, atol=1e-3)


def test_svd_lower_rank_has_larger_error() -> None:
    """Truncating SVD to fewer components increases reconstruction error."""
    k, _ = _make_cache()
    e_full = float(torch.norm(structured.svd_approx(k, 16) - k))
    e_half = float(torch.norm(structured.svd_approx(k, 8) - k))
    e_quarter = float(torch.norm(structured.svd_approx(k, 4) - k))
    assert e_full < e_half < e_quarter


def test_svd_reconstruction_beat_random_projection() -> None:
    """Rank-r SVD captures more energy than a random rank-r projection."""
    from run_kv_latent_compare import build_projection  # type: ignore

    k, _ = _make_cache()
    rank = HEAD_DIM // 2
    w_down, w_up = build_projection(
        HEAD_DIM,
        rank,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    proj_recon = k @ w_down @ w_up
    svd_recon = structured.svd_approx(k, rank)
    assert torch.norm(svd_recon - k) < torch.norm(proj_recon - k)


def test_pool_tokens_reduces_sequence() -> None:
    """Token pooling merges sequence positions by the pool size."""
    k, v = _make_cache(seq=16)
    past = ((k, v),)
    pooled = structured.pool_tokens_cache(past, 4)
    pk, pv = structured.to_legacy_kv(pooled)[0]
    assert pk.shape[2] == 4
    assert pv.shape[2] == 4


def test_pool_tokens_keeps_remainder() -> None:
    """Non-multiple sequences keep the trailing partial group."""
    k, _ = _make_cache(seq=18)
    pooled = structured.pool_seq(k, 4)
    assert pooled.shape[2] == 6  # 4 + 2 remainder


def test_effective_kv_bytes_reduces_with_compression() -> None:
    """Effective footprint shrinks with the compression ratio for every method."""
    stored = 1_000_000
    # head_prune and svd: a higher ratio directly shrinks the footprint.
    for method in ("head_prune", "svd"):
        small = structured.effective_kv_bytes(method, 4, NUM_HEADS, HEAD_DIM, stored)
        large = structured.effective_kv_bytes(method, 2, NUM_HEADS, HEAD_DIM, stored)
        assert small < large < stored
    # token_pool: the footprint is the pooled (shorter) cache's physical bytes,
    # so pooling a cache must report fewer effective bytes than an unpadded one.
    pooled_stored = stored // 4
    pooled_eff = structured.effective_kv_bytes("token_pool", 2, NUM_HEADS, HEAD_DIM, pooled_stored)
    assert pooled_eff == pooled_stored
    assert pooled_eff < stored


def test_json_output_exists() -> None:
    """At least one experiment JSON report exists after a structured run."""
    reports = _experiment_json_files()
    assert len(reports) >= 1, "No JSON reports found in reports/kv_structured_compare/"


def test_all_methods_and_ratios_covered() -> None:
    """Baseline plus every (method, ratio) config produced a report."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    seen: set[tuple[str, int]] = set()
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("method") != "baseline":
            seen.add((data.get("method"), data.get("compression_ratio")))

    expected = {("head_prune", 2), ("head_prune", 4), ("svd", 2), ("svd", 4), ("token_pool", 2), ("token_pool", 4)}
    assert expected <= seen, f"Missing configs: {expected - seen}"


def test_kv_size_reduces_with_compression() -> None:
    """Compressed runs must report fewer effective KV bytes than the baseline."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    baseline_kv: int | None = None
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("method") == "baseline":
            baseline_kv = int(data["kv_bytes"])
            break
    if baseline_kv is None:
        pytest.skip("No baseline report available")

    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("method") != "baseline":
            assert int(data["kv_bytes"]) < baseline_kv, (
                f"{path.name} did not reduce KV bytes"
            )


def test_outputs_diverge_from_baseline() -> None:
    """At least one compressed method must differ from the baseline output."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    baseline_out: str | None = None
    outputs: list[str] = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("method") == "baseline":
            baseline_out = data.get("output")
        else:
            outputs.append(data.get("output", ""))

    if baseline_out is None:
        pytest.skip("No baseline report available")
    assert any(o != baseline_out for o in outputs), "All outputs identical to baseline"


def test_report_schema_fields() -> None:
    """Each experiment JSON contains the required metric fields."""
    reports = _experiment_json_files()
    if not reports:
        pytest.skip("No experiment JSON files exist")

    required = {
        "method",
        "compression_ratio",
        "output",
        "tokens_generated",
        "latency_ms",
        "kv_bytes",
        "divergence",
        "break_point",
    }
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - set(data.keys())
        assert not missing, f"Missing fields in {path.name}: {missing}"
