"""Tests for reusable low-rank KV subspace decoding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Import helpers without executing run().
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_kv_subspace as builder  # noqa: E402
import run_kv_subspace_decode as decoder  # noqa: E402


BASIS_DIR = Path("reports/kv_subspace")
DECODE_DIR = Path("reports/kv_subspace_decode")

HEAD_DIM = 64
RANK = 16


def _model_basis_dir() -> Path:
    return BASIS_DIR / decoder.MODEL_NAME.split("/")[-1]


def _decode_report_files() -> list[Path]:
    if not DECODE_DIR.is_dir():
        return []
    return sorted(
        p
        for p in DECODE_DIR.glob("run_*.json")
        if p.is_file() and not p.name.startswith("run_summary_")
    )


def _make_cache(seq: int = 16, heads: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    k = torch.randn(1, heads, seq, HEAD_DIM)
    v = torch.randn(1, heads, seq, HEAD_DIM)
    return [(k, v)]


def test_script_modules_load() -> None:
    """Builder and decoder modules import cleanly."""
    assert hasattr(builder, "build_basis")
    assert hasattr(builder, "trace_cache")
    assert hasattr(decoder, "subspace_compress_cache")
    assert hasattr(decoder, "load_basis")
    assert hasattr(decoder, "greedy_generate")
    assert decoder.METHODS == ["baseline", "svd", "head_prune", "subspace"]


def test_build_basis_shapes_and_variance() -> None:
    """Basis has [layers, head_dim, rank] shape and variance in (0, 1]."""
    cache = builder.sc.from_legacy_kv(_make_cache(seq=24))
    w_k, w_v, ev_k, ev_v, proj_err = builder.build_basis(cache, RANK)
    assert w_k.shape == (1, HEAD_DIM, RANK)
    assert w_v.shape == (1, HEAD_DIM, RANK)
    assert 0.0 < ev_k <= 1.0
    assert 0.0 < ev_v <= 1.0
    assert proj_err >= 0.0

    # Full-rank basis should be (near) lossless.
    w_full, _, ev_full, _, perr_full = builder.build_basis(cache, HEAD_DIM)
    assert ev_full > ev_k
    assert perr_full < proj_err


def test_basis_columns_are_orthonormal() -> None:
    """Each basis matrix has orthonormal columns (W^T W == I)."""
    cache = builder.sc.from_legacy_kv(_make_cache(seq=24))
    w_k, _w_v, _a, _b, _c = builder.build_basis(cache, RANK)
    gram = w_k[0].t() @ w_k[0]
    assert torch.allclose(gram, torch.eye(RANK), atol=1e-5)


def test_subspace_compress_preserves_shape_and_is_idempotent() -> None:
    """Projection keeps tensor shapes and re-projecting changes nothing."""
    layers = _make_cache(seq=16)
    cache = builder.sc.from_legacy_kv(layers)
    w_k, w_v, *_ = builder.build_basis(cache, RANK)

    once = decoder.subspace_compress_cache(cache, w_k, w_v)
    twice = decoder.subspace_compress_cache(once, w_k, w_v)
    once_layers = builder.sc.to_legacy_kv(once)
    twice_layers = builder.sc.to_legacy_kv(twice)
    assert once_layers[0][0].shape == layers[0][0].shape
    assert torch.allclose(once_layers[0][0], twice_layers[0][0], atol=1e-5)


def test_subspace_reduces_effective_kv_bytes_by_ratio() -> None:
    """Effective KV bytes for subspace/svd scale as rank / head_dim."""
    stored = 1_000_000
    eff = decoder.effective_kv_bytes("subspace", stored, 4, HEAD_DIM, RANK)
    assert eff == stored * RANK // HEAD_DIM
    assert decoder.effective_kv_bytes("baseline", stored, 4, HEAD_DIM, RANK) == stored


def test_basis_file_and_meta_exist() -> None:
    """A basis.pt and basis_meta.json exist for the configured model."""
    dir_ = _model_basis_dir()
    assert (dir_ / "basis.pt").is_file(), "basis.pt missing; run build_kv_subspace.py"
    meta_path = dir_ / "basis_meta.json"
    assert meta_path.is_file(), "basis_meta.json missing"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for field in ("model", "rank", "head_dim", "explained_variance_k", "projection_error"):
        assert field in meta, f"Missing basis_meta field: {field}"


def test_basis_loads_from_disk() -> None:
    """Saved basis tensors load with the expected shape and rank."""
    dir_ = _model_basis_dir()
    if not (dir_ / "basis.pt").is_file():
        pytest.skip("No basis.pt on disk")
    w_k, w_v, rank = decoder.load_basis(
        decoder.MODEL_NAME,
        torch.device("cpu"),
        torch.float32,
    )
    assert rank == decoder.RANK_DIVISOR or rank > 0
    assert w_k.shape[-1] == rank
    assert w_v.shape[-1] == rank


def test_decode_reports_exist() -> None:
    """At least one per-method decode report exists."""
    reports = _decode_report_files()
    assert len(reports) >= 1, "No run_*.json in reports/kv_subspace_decode/"


def test_all_methods_covered() -> None:
    """Baseline, svd, head_prune, subspace each produced a report."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No decode reports exist")
    methods = {json.loads(p.read_text(encoding="utf-8"))["method"] for p in reports}
    assert methods == set(decoder.METHODS), f"Missing methods: {set(decoder.METHODS) - methods}"


def test_report_schema() -> None:
    """Each decode report carries the required metric fields."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No decode reports exist")
    required = {
        "method",
        "rank",
        "kv_bytes",
        "latency_ms",
        "divergence",
        "recon_error",
    }
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - set(data.keys())
        assert not missing, f"Missing fields in {path.name}: {missing}"


def test_divergence_is_measured() -> None:
    """Every non-baseline report records a break point and divergence text."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No decode reports exist")
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "break_point" in data, f"Missing break_point in {path.name}"
        assert isinstance(data["divergence"], str) and data["divergence"]
        assert isinstance(data["recon_error"], float)


def test_kv_reduction_matches_expected_ratio() -> None:
    """Subspace and svd reduce effective KV by ~rank/head_dim vs baseline."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No decode reports exist")
    baseline_kv: int | None = None
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["method"] == "baseline":
            baseline_kv = int(data["kv_bytes"])
            break
    if baseline_kv is None:
        pytest.skip("No baseline report available")

    ratio = RANK / HEAD_DIM
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["method"] in ("subspace", "svd"):
            expected = baseline_kv * ratio
            assert int(data["kv_bytes"]) <= expected * 1.05 + 1, (
                f"{data['method']} kv_bytes not ~rank/head_dim of baseline"
            )
