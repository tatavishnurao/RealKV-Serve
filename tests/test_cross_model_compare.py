"""Tests for cross-model KV-cache property validation reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import helpers from the cross-model runner without executing run().
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cross_model_compare as cm  # noqa: E402


REPORT_DIR = Path("reports/cross_model")


def _model_json_files() -> list[Path]:
    """Per-model report files (exclude summary_*)."""
    if not REPORT_DIR.is_dir():
        return []
    return sorted(
        p
        for p in REPORT_DIR.glob("*.json")
        if p.is_file() and not p.name.startswith("summary_")
    )


def _summary_files() -> list[Path]:
    if not REPORT_DIR.is_dir():
        return []
    return sorted(REPORT_DIR.glob("summary_*.json"))


def test_script_module_loads() -> None:
    """Cross-model runner module imports cleanly (no crash on import)."""
    assert hasattr(cm, "run")
    assert hasattr(cm, "process_model")
    assert hasattr(cm, "trace_decode")
    assert hasattr(cm, "detect_trends")
    assert len(cm.MODELS) >= 3


def test_report_files_exist() -> None:
    """At least one per-model JSON report exists after a cross-model run."""
    reports = _model_json_files()
    assert len(reports) >= 1, "No JSON reports found in reports/cross_model/"


def test_every_configured_model_has_a_report() -> None:
    """Each configured model produced a report (available or skipped)."""
    reports = _model_json_files()
    if not reports:
        pytest.skip("No per-model reports exist")

    slugs = {p.name.split("_")[0] for p in reports}
    configured = {name.split("/")[-1] for name in cm.MODELS}
    assert configured.issubset(slugs), f"Missing model reports: {configured - slugs}"


def test_summary_exists_and_schema_valid() -> None:
    """A summary JSON exists with the required top-level keys."""
    summaries = _summary_files()
    if not summaries:
        pytest.skip("No summary report exists")

    data = json.loads(summaries[-1].read_text(encoding="utf-8"))
    required = {
        "device",
        "prompt",
        "max_new_tokens",
        "table_columns",
        "table_rows",
        "trends",
        "models",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing summary fields: {missing}"


def test_summary_table_rows_match_available_models() -> None:
    """Summary table has one row per available model, with all columns."""
    summaries = _summary_files()
    if not summaries:
        pytest.skip("No summary report exists")

    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    avail = [m for m in summary["models"] if m.get("available")]
    assert len(summary["table_rows"]) == len(avail), (
        "Summary rows must match the number of available models"
    )
    assert len(summary["table_columns"]) == 11


def test_model_report_schema_valid() -> None:
    """Every per-model report carries the required metric fields."""
    reports = _model_json_files()
    if not reports:
        pytest.skip("No per-model reports exist")

    required = {
        "model",
        "available",
        "parameters",
        "layers",
        "kv_heads",
        "head_dim",
        "svd_error",
        "head_prune_error",
        "runs",
        "skipped_reason",
    }
    skip_required = {"model", "available", "skipped_reason"}
    run_required = {
        "kv_bytes",
        "kv_bytes_stored",
        "bytes_per_token",
        "decode_latency_ms",
        "prefill_latency_ms",
        "output",
        "tokens_generated",
    }
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("available"):
            missing = skip_required - set(data.keys())
            assert not missing, f"Missing skip fields in {path.name}: {missing}"
            continue
        missing = required - set(data.keys())
        assert not missing, f"Missing fields in {path.name}: {missing}"
        for name, run in data["runs"].items():
            rmissing = run_required - set(run.keys())
            assert not rmissing, f"Missing run fields in {path.name}/{name}: {rmissing}"


def test_skipped_models_include_reason() -> None:
    """Unavailable models record an explicit skipped_reason."""
    reports = _model_json_files()
    if not reports:
        pytest.skip("No per-model reports exist")

    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data["available"]:
            assert data.get("skipped_reason"), f"Missing skipped_reason in {path.name}"


def test_trends_are_data_driven_statements() -> None:
    """Trends are non-empty strings produced by the detector."""
    summaries = _summary_files()
    if not summaries:
        pytest.skip("No summary report exists")

    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    assert isinstance(summary["trends"], list)
    assert all(isinstance(t, str) and t for t in summary["trends"])
