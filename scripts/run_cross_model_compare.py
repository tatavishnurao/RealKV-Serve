"""Cross-model KV-cache property validation across real open-weight LLMs.

Loads every available model from a fixed list, runs an identical prompt through
the same greedy decode loop under baseline, structured SVD, and head-pruning
compression, and records KV footprint, bytes/token, prefill/decode latency, GPU
memory, and per-layer structure metrics.

Models that cannot be downloaded or do not fit in VRAM are skipped
automatically with a reason. Compression implementations are reused from
`run_kv_structured_compare.py`; no new compression methods, serving
infrastructure, batching, or distributed inference are introduced.

Reproducible empirical evidence only: no hardcoded conclusions.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_kv_structured_compare import (  # noqa: E402
    apply_compression,
    effective_kv_bytes,
    kv_cache_bytes,
    resolve_keep_heads,
    resolve_rank,
    svd_approx,
    to_legacy_kv,
    warmup,
)

DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
REPORT_DIR = Path("reports/cross_model")
COMPRESSION_CONFIGS = [
    ("svd", 2),
    ("svd", 4),
    ("head_prune", 2),
    ("head_prune", 4),
]

MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-2-2b-it",
]


def model_heads(model) -> int:
    """KV-head count for the model (GQA-aware)."""
    return int(
        getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    )


def model_head_dim(model) -> int:
    """Per-head feature dimension for the model."""
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    return int(head_dim)


def trace_decode(
    model,
    tokenizer,
    prompt: str,
    device: str,
    method: str,
    compression_ratio: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Prefill + greedy decode under one KV strategy; returns metrics + cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []
    bpt_samples: list[float] = []

    num_heads = model_heads(model)
    head_dim = model_head_dim(model)

    with torch.no_grad():
        prefill_start = time.monotonic()
        out = model(input_ids=generated, use_cache=True, past_key_values=None)
        prefill_ms = (time.monotonic() - prefill_start) * 1000.0

    logits = out.logits[:, -1, :]
    next_token = logits.argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    past_key_values = out.past_key_values
    if method != "baseline":
        past_key_values = apply_compression(
            past_key_values,
            method,
            compression_ratio,
            num_heads,
            head_dim,
        )

    def bytes_per_token(pkv: Any) -> float:
        layers = to_legacy_kv(pkv)
        if not layers:
            return 0.0
        return kv_cache_bytes(pkv) / layers[0][0].shape[2]

    for step in range(max_new_tokens - 1):
        step_start = time.monotonic()
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        past_key_values = out.past_key_values
        if method != "baseline":
            past_key_values = apply_compression(
                past_key_values,
                method,
                compression_ratio,
                num_heads,
                head_dim,
            )
        latencies.append((time.monotonic() - step_start) * 1000.0)
        if method == "baseline" and step % 8 == 0:
            bpt_samples.append(bytes_per_token(past_key_values))

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stored_bytes = kv_cache_bytes(past_key_values)
    seq_len = to_legacy_kv(past_key_values)[0][0].shape[2]

    return {
        "cache": past_key_values,
        "kv_bytes": effective_kv_bytes(
            method,
            compression_ratio,
            num_heads,
            head_dim,
            stored_bytes,
        ),
        "kv_bytes_stored": stored_bytes,
        "seq_len": seq_len,
        "bytes_per_token": round(stored_bytes / seq_len, 1) if seq_len else 0,
        "bytes_per_token_samples": [round(b, 1) for b in bpt_samples],
        "decode_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "prefill_latency_ms": round(prefill_ms, 3),
        "tokens_generated": int(new_tokens.shape[0]),
        "token_ids": [int(t) for t in new_tokens.tolist()],
        "output": output_text,
    }


def mean_svd_error(cache: Any, rank: int) -> float:
    """Mean relative SVD reconstruction error over layers (K and V)."""
    layers = to_legacy_kv(cache)
    errs: list[float] = []
    for k, v in layers:
        ek = torch.norm(svd_approx(k, rank).float() - k.float()) / torch.norm(k.float())
        ev = torch.norm(svd_approx(v, rank).float() - v.float()) / torch.norm(v.float())
        errs.append((float(ek) + float(ev)) / 2.0)
    return sum(errs) / len(errs) if errs else 0.0


def mean_head_prune_error(cache: Any, num_heads: int, ratio: int) -> float:
    """Mean relative K/V error after zeroing pruned heads."""
    keep = resolve_keep_heads(num_heads, ratio)
    mask = torch.zeros(num_heads, dtype=torch.bool)
    mask[:keep] = True
    mask = mask.view(1, -1, 1, 1)
    layers = to_legacy_kv(cache)
    errs: list[float] = []
    for k, v in layers:
        km = mask.to(k.device)
        ek = torch.norm(k.masked_fill(~km, 0).float() - k.float()) / torch.norm(k.float())
        ev = torch.norm(v.masked_fill(~km, 0).float() - v.float()) / torch.norm(v.float())
        errs.append((float(ek) + float(ev)) / 2.0)
    return sum(errs) / len(errs) if errs else 0.0


def svd_energy_metrics(cache: Any) -> tuple[float, float]:
    """(participation ratio, rank retaining 90% of Frobenius energy) over layers."""
    layers = to_legacy_kv(cache)
    prs: list[float] = []
    ranks: list[float] = []
    for k, _v in layers:
        _u, s, _vh = torch.linalg.svd(k.float(), full_matrices=False)
        s2 = s**2
        total = torch.clamp(s2.sum(dim=-1, keepdim=True), min=1e-12)
        norm_s2 = s2 / total
        prs.append(float((1.0 / (norm_s2**2).sum(dim=-1)).mean()))
        cum = torch.cumsum(norm_s2, dim=-1)
        ranks.append(float(((cum < 0.90).sum(dim=-1) + 1).float().mean()))
    return (sum(prs) / len(prs), sum(ranks) / len(ranks)) if prs else (0.0, 0.0)


def describe_divergence(baseline: str, other: str) -> str:
    """Human-readable divergence summary vs baseline text."""
    if baseline == other:
        return "identical"
    common_prefix = 0
    for a, b in zip(baseline, other):
        if a != b:
            break
        common_prefix += 1
    return (
        f"differs (len_base={len(baseline)}, len_other={len(other)}, "
        f"common_prefix_chars={common_prefix})"
    )


def token_break_point(baseline_ids: list[int], other_ids: list[int]) -> int | None:
    """Index of the first generated token that differs from baseline."""
    for i, (a, b) in enumerate(zip(baseline_ids, other_ids)):
        if a != b:
            return i
    return None


def coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation (std / mean); 1.0 if degenerate."""
    if not values:
        return 1.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 1.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (var**0.5) / mean


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient; 0.0 on degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def process_model(model_name: str, device: str, max_new_tokens: int) -> dict[str, Any]:
    """Run the full per-model experiment; never raises for a single model."""
    print(f"\n===== {model_name} =====")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            use_cache=True,
        ).to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model_name,
            "available": False,
            "skipped_reason": f"load failed: {exc}",
        }

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        warmup(model, tokenizer, device)

        num_heads = model_heads(model)
        head_dim = model_head_dim(model)
        num_layers = int(model.config.num_hidden_layers)
        params = model.num_parameters()

        baseline = trace_decode(model, tokenizer, DEFAULT_PROMPT, device, "baseline", 1)
        baseline_cache = baseline.pop("cache")
        baseline_ids: list[int] = baseline["token_ids"]

        runs: dict[str, Any] = {"baseline": baseline}
        for method, ratio in COMPRESSION_CONFIGS:
            res = trace_decode(
                model,
                tokenizer,
                DEFAULT_PROMPT,
                device,
                method,
                ratio,
            )
            res.pop("cache")
            res["break_point"] = token_break_point(baseline_ids, res["token_ids"])
            res["divergence"] = describe_divergence(baseline["output"], res["output"])
            runs[f"{method}_ratio{ratio}"] = res

        svd_error_r2 = mean_svd_error(baseline_cache, resolve_rank(head_dim, 2))
        svd_error_r4 = mean_svd_error(baseline_cache, resolve_rank(head_dim, 4))
        prune_error_r2 = mean_head_prune_error(baseline_cache, num_heads, 2)
        prune_error_r4 = mean_head_prune_error(baseline_cache, num_heads, 4)
        eff_rank, rank90 = svd_energy_metrics(baseline_cache)

        gpu_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        )

        return {
            "model": model_name,
            "available": True,
            "parameters": int(params),
            "layers": num_layers,
            "kv_heads": num_heads,
            "head_dim": head_dim,
            "gpu_memory_mb": round(gpu_mb, 1),
            "svd_rank": resolve_rank(head_dim, 4),
            "svd_error": round(svd_error_r4, 4),
            "svd_error_r2": round(svd_error_r2, 4),
            "head_prune_error": round(prune_error_r4, 4),
            "head_prune_error_r2": round(prune_error_r2, 4),
            "effective_rank": round(eff_rank, 2),
            "rank90": round(rank90, 2),
            "rank90_norm": round(rank90 / head_dim, 3),
            "bytes_per_token_cv": round(coefficient_of_variation(baseline["bytes_per_token_samples"]), 4),
            "svd_output_identical": all(
                runs[f"svd_ratio{r}"]["break_point"] is None for r in (2, 4)
            ),
            "runs": runs,
            "skipped_reason": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model_name,
            "available": False,
            "skipped_reason": f"experiment failed: {exc}",
        }
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_report(result: dict[str, Any], path: Path) -> None:
    """Write one structured JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


def detect_trends(model_data: list[dict[str, Any]]) -> list[str]:
    """Automated data-driven trend statements (no hardcoded conclusions)."""
    avail = [d for d in model_data if d.get("available")]
    trends: list[str] = []
    if len(avail) < 2:
        trends.append("Fewer than 2 models available; trend analysis skipped.")
        return trends

    linear_ok = all(d["bytes_per_token_cv"] < 0.20 for d in avail)
    trends.append(
        "KV bytes grow ~linearly with sequence length across models "
        f"(bytes/token CV per model < 0.20: {linear_ok}; "
        f"observed CVs={[d['bytes_per_token_cv'] for d in avail]})"
    )

    r90 = [d["rank90_norm"] for d in avail]
    r90_cv = coefficient_of_variation(r90)
    trends.append(
        "SVD 90% energy-retention rank (fraction of head_dim): "
        f"range {min(r90):.3f}-{max(r90):.3f}, CV={r90_cv:.3f} -> "
        f"{'similar across models' if r90_cv < 0.35 else 'divergent across models'}"
    )

    svd_ok = [d["svd_output_identical"] for d in avail]
    trends.append(
        f"SVD reconstruction preserved baseline output exactly: "
        f"{sum(svd_ok)}/{len(avail)} models"
    )

    svd_better = sum(1 for d in avail if d["svd_error"] < d["head_prune_error"])
    trends.append(
        f"SVD error < head-prune error on {svd_better}/{len(avail)} models "
        f"(svd_errors={[d['svd_error'] for d in avail]}, "
        f"prune_errors={[d['head_prune_error'] for d in avail]})"
    )

    params = [float(d["parameters"]) for d in avail]
    kv = [float(d["runs"]["baseline"]["kv_bytes_stored"]) for d in avail]
    lats = [float(d["runs"]["baseline"]["decode_latency_ms"]) for d in avail]
    prefs = [float(d["runs"]["baseline"]["prefill_latency_ms"]) for d in avail]
    r_param_kv = pearson(params, kv)
    r_param_lat = pearson(params, lats)
    r_param_pref = pearson(params, prefs)
    trends.append(
        "Parameter count correlations: r(KV bytes)="
        f"{r_param_kv:.2f}, r(decode latency)={r_param_lat:.2f}, "
        f"r(prefill latency)={r_param_pref:.2f}"
    )
    return trends


def build_table(avail: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """Summary table columns + rows over available models."""
    header = [
        "Model",
        "Parameters",
        "Layers",
        "KV Heads",
        "Head Dim",
        "KV Bytes",
        "Bytes/Token",
        "SVD Rank",
        "SVD Error",
        "Head Prune Error",
        "Latency (ms)",
    ]
    rows: list[list[Any]] = []
    for d in avail:
        b = d["runs"]["baseline"]
        rows.append(
            [
                d["model"].split("/")[-1],
                f"{d['parameters'] / 1e9:.2f}B",
                d["layers"],
                d["kv_heads"],
                d["head_dim"],
                f"{b['kv_bytes_stored'] / 1e6:.2f}M",
                f"{b['bytes_per_token'] / 1e3:.1f}K",
                d["svd_rank"],
                f"{d['svd_error']:.3f}",
                f"{d['head_prune_error']:.3f}",
                f"{b['decode_latency_ms']:.1f}",
            ]
        )
    return header, rows


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    model_data: list[dict[str, Any]] = []
    for name in MODELS:
        data = process_model(name, device, MAX_NEW_TOKENS)
        model_data.append(data)
        slug = name.split("/")[-1]
        write_report(data, REPORT_DIR / f"{slug}_{stamp}.json")
        if data.get("available"):
            b = data["runs"]["baseline"]
            print(
                f"[ok] {name}: kv_bytes={b['kv_bytes_stored']}, "
                f"bytes/token={b['bytes_per_token']}, "
                f"svd_error={data['svd_error']}, "
                f"decode_ms={b['decode_latency_ms']}"
            )
        else:
            print(f"[skip] {name}: {data['skipped_reason']}")

    avail = [d for d in model_data if d.get("available")]
    trends = detect_trends(model_data)
    header, rows = build_table(avail)

    print("\n=== Summary table ===")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")

    print("\n=== Detected trends ===")
    for t in trends:
        print("- " + t)

    summary = {
        "device": device,
        "prompt": DEFAULT_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "timestamp": stamp,
        "table_columns": header,
        "table_rows": rows,
        "trends": trends,
        "models": model_data,
    }
    write_report(summary, REPORT_DIR / f"summary_{stamp}.json")

    print(f"\nReports written under {REPORT_DIR}/")
    print("CROSS_MODEL_COMPARE_OK=1")


if __name__ == "__main__":
    run()
