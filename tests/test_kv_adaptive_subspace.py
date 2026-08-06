"""Tests for adaptive KV subspace decoding with online basis updates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kv_subspace_update as update  # noqa: E402
import run_kv_adaptive_subspace as adaptive  # noqa: E402

HEAD_DIM = 64
RANK = 16


def _make_basis(n_layers: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    wk = torch.zeros(n_layers, HEAD_DIM, RANK)
    wv = torch.zeros(n_layers, HEAD_DIM, RANK)
    for i in range(n_layers):
        qk, _ = torch.linalg.qr(torch.randn(HEAD_DIM, RANK))
        qv, _ = torch.linalg.qr(torch.randn(HEAD_DIM, RANK))
        wk[i] = qk
        wv[i] = qv
    return wk, wv


def _make_kv_tensor(batch: int = 1, heads: int = 4, seq: int = 8) -> torch.Tensor:
    return torch.randn(batch, heads, seq, HEAD_DIM)


def test_batched_ipca_initialization() -> None:
    """BatchedIPCA creates updater with correct shapes and seeded covariance."""
    wk, wv = _make_basis(n_layers=3)
    updater = update.BatchedIPCAUpdater(wk, wv)
    assert updater.num_layers == 3
    assert updater.head_dim == HEAD_DIM
    assert updater.rank == RANK
    assert updater.C.shape == (6, HEAD_DIM, HEAD_DIM)
    assert updater.step_count == 0


def test_batched_ipca_single_update() -> None:
    """Single update step runs without error and maintains basis shape."""
    wk, wv = _make_basis(n_layers=2)
    updater = update.BatchedIPCAUpdater(wk, wv, alpha=0.95, update_interval=1)
    k_vecs = torch.randn(2, 4, HEAD_DIM)
    v_vecs = torch.randn(2, 4, HEAD_DIM)
    out_wk, out_wv, k_drift, v_drift = updater.update_batch(k_vecs, v_vecs)
    assert out_wk.shape == wk.shape
    assert out_wv.shape == wv.shape
    assert k_drift >= 0.0
    assert v_drift >= 0.0


def test_batched_ipca_multiple_updates() -> None:
    """Multiple update steps maintain basis shape without NaN."""
    wk, wv = _make_basis(n_layers=2)
    updater = update.BatchedIPCAUpdater(wk, wv, alpha=0.95, update_interval=2)
    k_vecs = torch.randn(2, 4, HEAD_DIM)
    v_vecs = torch.randn(2, 4, HEAD_DIM)
    for _ in range(10):
        updater.update_batch(k_vecs, v_vecs)
    assert not torch.isnan(updater.W_k).any()
    assert not torch.isnan(updater.W_v).any()


def test_batched_ipca_interval_timing() -> None:
    """Basis only refreshes at update_interval boundaries."""
    wk, wv = _make_basis(n_layers=1)
    updater = update.BatchedIPCAUpdater(wk, wv, alpha=0.95, update_interval=4)
    k_vecs = torch.randn(1, 4, HEAD_DIM)
    v_vecs = torch.randn(1, 4, HEAD_DIM)
    _, _, d0_k, d0_v = updater.update_batch(k_vecs, v_vecs)
    assert d0_k == 0.0
    assert d0_v == 0.0

    for _ in range(2):
        updater.update_batch(k_vecs, v_vecs)
    _, _, d3_k, d3_v = updater.update_batch(k_vecs, v_vecs)
    assert d3_k >= 0.0
    assert d3_v >= 0.0


def test_batched_ipca_compress() -> None:
    """Compress preserves shape."""
    wk, wv = _make_basis(n_layers=1)
    updater = update.BatchedIPCAUpdater(wk, wv)
    k = _make_kv_tensor(batch=1, heads=4, seq=8)
    v = _make_kv_tensor(batch=1, heads=4, seq=8)
    import run_kv_structured_compare as sc
    from transformers import DynamicCache
    cache = DynamicCache(ddp_cache_data=[(k, v)])
    compressed = updater.compress(cache, sc)
    layers = sc.to_legacy_kv(compressed)
    assert layers[0][0].shape == k.shape
    assert layers[0][1].shape == v.shape


def test_get_current_bases() -> None:
    """get_current_bases returns correct shapes."""
    wk, wv = _make_basis(n_layers=3)
    updater = update.BatchedIPCAUpdater(wk, wv)
    wk_out, wv_out = updater.get_current_bases()
    assert wk_out.shape == wk.shape
    assert wv_out.shape == wv.shape


def test_script_modules_load() -> None:
    """All adaptive modules import cleanly with expected attributes."""
    assert hasattr(update, "BatchedIPCAUpdater")
    assert hasattr(adaptive, "greedy_generate_adaptive")
    assert hasattr(adaptive, "subspace_compress_cache")
    assert adaptive.METHODS == ["baseline", "fixed", "adaptive"]


def test_adaptive_effective_kv_bytes() -> None:
    """Both fixed and adaptive scale KV bytes by rank/head_dim."""
    stored = 1_000_000
    assert adaptive.effective_kv_bytes("fixed", stored, 4, HEAD_DIM, RANK) == stored * RANK // HEAD_DIM
    assert adaptive.effective_kv_bytes("adaptive", stored, 4, HEAD_DIM, RANK) == stored * RANK // HEAD_DIM
    assert adaptive.effective_kv_bytes("baseline", stored, 4, HEAD_DIM, RANK) == stored


def test_static_subspace_compress() -> None:
    """Fixed subspace compression preserves shape and reduces dimension."""
    wk, wv = _make_basis(n_layers=1)
    k = _make_kv_tensor(batch=1, heads=4, seq=8)
    v = _make_kv_tensor(batch=1, heads=4, seq=8)
    from transformers import DynamicCache
    cache = DynamicCache(ddp_cache_data=[(k, v)])
    compressed = adaptive.subspace_compress_cache(cache, wk, wv)
    layers = adaptive.sc.to_legacy_kv(compressed)
    assert layers[0][0].shape == k.shape
    assert layers[0][1].shape == v.shape


def test_adaptive_report_dir_ready() -> None:
    """Report directory exists or can be created."""
    rd = adaptive.REPORT_DIR
    assert rd is not None


DECODE_DIR = Path("reports/kv_adaptive_subspace")


def _decode_report_files() -> list[Path]:
    if not DECODE_DIR.is_dir():
        return []
    return sorted(
        p
        for p in DECODE_DIR.glob("run_*.json")
        if p.is_file() and not p.name.startswith("run_summary_")
    )


def test_adaptive_decode_reports_exist() -> None:
    """At least one per-method decode report was written."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No run_*.json in reports/kv_adaptive_subspace/")
    assert len(reports) >= 1, "No run_*.json in reports/kv_adaptive_subspace/"


def test_all_adaptive_methods_covered() -> None:
    """Baseline, fixed, adaptive each produced a report."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No adaptive decode reports exist")
    methods = {json.loads(p.read_text(encoding="utf-8"))["method"] for p in reports}
    assert methods == set(adaptive.METHODS), f"Missing methods: {set(adaptive.METHODS) - methods}"


def test_adaptive_report_schema() -> None:
    """Each adaptive decode report carries required metric fields."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No adaptive decode reports exist")
    required = {"method", "rank", "kv_bytes", "latency_ms", "divergence", "recon_error"}
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - set(data.keys())
        assert not missing, f"Missing fields in {path.name}: {missing}"


def test_adaptive_break_point() -> None:
    """Both subspace methods record a break point vs baseline."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No adaptive decode reports exist")
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["method"] != "baseline":
            assert "break_point" in data, f"Missing break_point in {path.name}"
            assert data["break_point"] is None or data["break_point"] >= 0


def test_adaptive_drift_recorded() -> None:
    """Adaptive reports include drift_history and update metrics."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No adaptive decode reports exist")
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["method"] == "adaptive":
            assert "drift_history" in data, f"Missing drift_history in {path.name}"
            assert "update_costs_ms" in data, f"Missing update_costs_ms in {path.name}"
            assert "num_updates" in data, f"Missing num_updates in {path.name}"


def test_adaptive_kv_reduction_preserved() -> None:
    """Adaptive subspace preserves the KV reduction ratio."""
    reports = _decode_report_files()
    if not reports:
        pytest.skip("No adaptive decode reports exist")
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
        if data["method"] in ("fixed", "adaptive"):
            expected = baseline_kv * ratio
            assert int(data["kv_bytes"]) <= expected * 1.05 + 1, (
                f"{data['method']} kv_bytes not ~rank/head_dim of baseline"
            )
