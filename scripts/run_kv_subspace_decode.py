"""Runtime low-rank KV decoding reusing a precomputed subspace basis.

Loads `reports/kv_subspace/<model>/basis.pt` (built by build_kv_subspace.py)
and compares four strategies inside the same greedy decode loop:

- baseline: full KV
- svd:      full truncated-SVD low-rank recomputed every step (slow reference)
- head_prune: keep 1/4 of kv-heads
- subspace: project KV into the fixed precomputed basis and reconstruct
            before attention (the reusable mechanism)

Collects latency, KV memory, output divergence (break point), and
reconstruction error vs the baseline cache. No batching, no serving infra.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_kv_structured_compare as sc

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
REPORT_DIR = Path("reports/kv_subspace_decode")
BASIS_ROOT = Path("reports/kv_subspace")
RANK_DIVISOR = 4

METHODS = ["baseline", "svd", "head_prune", "subspace"]


def load_model(device: str):
    """Load TinyLlama causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        use_cache=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def model_head_dim(model) -> int:
    """Per-head feature dimension for the model."""
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    return int(head_dim)


def model_num_heads(model) -> int:
    """KV-head count for the model (GQA-aware)."""
    return int(
        getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    )


def load_basis(model_name: str, device: torch.device, dtype: torch.dtype):
    """Load precomputed subspace basis; error clearly if missing."""
    slug = model_name.split("/")[-1]
    path = BASIS_ROOT / slug / "basis.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Basis not found at {path}. Run scripts/build_kv_subspace.py first."
        )
    data = torch.load(path, map_location="cpu")
    w_k = data["W_k"].to(device=device, dtype=dtype)
    w_v = data["W_v"].to(device=device, dtype=dtype)
    return w_k, w_v, int(data["rank"])


def subspace_compress_cache(
    past_key_values: Any,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
) -> Any:
    """Project every layer's K/V into the fixed basis and reconstruct."""
    layers = sc.to_legacy_kv(past_key_values)
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (k, v) in enumerate(layers):
        k_recon = (k @ w_k[i]) @ w_k[i].t()
        v_recon = (v @ w_v[i]) @ w_v[i].t()
        rebuilt.append((k_recon.contiguous(), v_recon.contiguous()))
    return sc.from_legacy_kv(rebuilt)


def effective_kv_bytes(
    method: str,
    stored_bytes: int,
    num_heads: int,
    head_dim: int,
    rank: int,
) -> int:
    """Effective on-disk footprint a real compressed-KV system would pay."""
    if method == "head_prune":
        keep = sc.resolve_keep_heads(num_heads, RANK_DIVISOR)
        return int(stored_bytes * keep / num_heads)
    if method in ("svd", "subspace"):
        return int(stored_bytes * rank / head_dim)
    return int(stored_bytes)


def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    method: str,
    rank: int,
    w_k: torch.Tensor | None = None,
    w_v: torch.Tensor | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Greedy decode under one KV strategy; returns metrics plus the cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []

    num_heads = model_num_heads(model)
    head_dim = model_head_dim(model)

    for step in range(max_new_tokens):
        step_start = time.monotonic()
        with torch.no_grad():
            if past_key_values is None:
                out = model(
                    input_ids=generated,
                    use_cache=True,
                    past_key_values=None,
                )
            else:
                out = model(
                    input_ids=generated[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values,
                )
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        past_key_values = out.past_key_values

        if method == "svd":
            past_key_values = sc.svd_compress_cache(past_key_values, rank)
        elif method == "head_prune":
            keep = sc.resolve_keep_heads(num_heads, RANK_DIVISOR)
            past_key_values = sc.head_prune_cache(past_key_values, keep, num_heads)
        elif method == "subspace":
            past_key_values = subspace_compress_cache(past_key_values, w_k, w_v)

        latencies.append((time.monotonic() - step_start) * 1000)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stored_bytes = sc.kv_cache_bytes(past_key_values)
    seq_len = sc.to_legacy_kv(past_key_values)[0][0].shape[2]
    return {
        "cache": past_key_values,
        "method": method,
        "rank": rank,
        "output": output_text,
        "tokens_generated": int(new_tokens.shape[0]),
        "token_ids": [int(t) for t in new_tokens.tolist()],
        "latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "kv_bytes": effective_kv_bytes(
            method,
            stored_bytes,
            num_heads,
            head_dim,
            rank,
        ),
        "kv_bytes_stored": stored_bytes,
        "bytes_per_token": round(stored_bytes / seq_len, 1) if seq_len else 0,
    }


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


def recon_error(cache_a: Any, cache_b: Any) -> float:
    """Mean relative K/V reconstruction error of cache_a vs cache_b."""
    layers_a = sc.to_legacy_kv(cache_a)
    layers_b = sc.to_legacy_kv(cache_b)
    errs: list[float] = []
    for (ka, va), (kb, vb) in zip(layers_a, layers_b):
        ek = torch.norm(ka.float() - kb.float()) / torch.norm(kb.float())
        ev = torch.norm(va.float() - vb.float()) / torch.norm(vb.float())
        errs.append((float(ek) + float(ev)) / 2.0)
    return sum(errs) / len(errs) if errs else 0.0


def write_report(result: dict[str, Any], path: Path) -> None:
    """Write one structured JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    model, tokenizer = load_model(device)
    sc.warmup(model, tokenizer, device)

    head_dim = model_head_dim(model)
    rank = max(1, head_dim // RANK_DIVISOR)

    print(f"Loading subspace basis (rank={rank}) ...")
    w_k, w_v, basis_rank = load_basis(MODEL_NAME, model.device, next(model.parameters()).dtype)
    print(f"Basis loaded: layers={w_k.shape[0]}, rank={basis_rank}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    baseline = greedy_generate(model, tokenizer, DEFAULT_PROMPT, device, "baseline", rank)
    baseline_cache = baseline.pop("cache")
    baseline_ids: list[int] = baseline["token_ids"]
    baseline["divergence"] = "identical"
    baseline["break_point"] = None
    baseline["recon_error"] = 0.0
    baseline["kv_saved_bytes"] = 0
    write_report(baseline, REPORT_DIR / f"run_baseline_{stamp}.json")
    results.append(baseline)
    print(f"\nbaseline   : '{baseline['output'][:60]}...'  "
          f"kv={baseline['kv_bytes_stored']}  lat={baseline['latency_ms']}ms")

    for method in ["svd", "head_prune", "subspace"]:
        print(f"\n=== {method} (rank/keep={rank}) ===")
        kwargs: dict[str, Any] = {}
        if method == "subspace":
            kwargs = {"w_k": w_k, "w_v": w_v}
        result = greedy_generate(
            model,
            tokenizer,
            DEFAULT_PROMPT,
            device,
            method,
            rank,
            **kwargs,
        )
        result_cache = result.pop("cache")
        result["divergence"] = describe_divergence(baseline["output"], result["output"])
        result["break_point"] = token_break_point(baseline_ids, result["token_ids"])
        result["recon_error"] = recon_error(result_cache, baseline_cache)
        result["kv_saved_bytes"] = baseline["kv_bytes_stored"] - result["kv_bytes"]
        write_report(result, REPORT_DIR / f"run_{method}_{stamp}.json")
        results.append(result)
        print(f"Output: {result['output'][:70]!r}")
        print(
            f"latency={result['latency_ms']}ms  kv_bytes={result['kv_bytes']}  "
            f"(stored {result['kv_bytes_stored']})  "
            f"break_point={result['break_point']}  recon_error={result['recon_error']:.4f}"
        )

    summary = {
        "model": MODEL_NAME,
        "device": device,
        "prompt": DEFAULT_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "rank": rank,
        "timestamp": stamp,
        "results": results,
    }
    write_report(summary, REPORT_DIR / f"run_summary_{stamp}.json")

    print(f"\nReports written under {REPORT_DIR}/")
    print("KV_SUBSPACE_DECODE_OK=1")


if __name__ == "__main__":
    run()
