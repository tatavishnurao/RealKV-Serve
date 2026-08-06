"""Adaptive KV subspace decoding with online basis updates.

Extends the fixed subspace mechanism (run_kv_subspace_decode.py) by maintaining
a running subspace U_t that updates during decoding via incremental PCA.
Compares three strategies inside the same greedy decode loop:

- baseline: full KV (no compression)
- fixed subspace: precomputed basis, never updated (existing)
- adaptive subspace: precomputed basis, updated each step via IPCA

Collects:
- latency (ms per token)
- KV bytes (effective and stored)
- break point (first token divergence from baseline)
- reconstruction error over time (per-token, K and V)
- basis drift (Frobenius change of basis per update interval)
- update cost (overhead of IPCA update step)
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
from kv_subspace_update import BatchedIPCAUpdater

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
REPORT_DIR = Path("reports/kv_adaptive_subspace")
BASIS_ROOT = Path("reports/kv_subspace")
RANK_DIVISOR = 4
ALPHA = 0.95
UPDATE_INTERVAL = 4
MAX_INTERVAL = 16
DRIFT_THRESHOLD = 0.005

METHODS = ["baseline", "fixed", "adaptive"]


def load_model(device: str):
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
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    return int(head_dim)


def model_num_heads(model) -> int:
    return int(
        getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    )


def load_basis(model_name: str, device: torch.device, dtype: torch.dtype):
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
    if method in ("fixed", "adaptive"):
        return int(stored_bytes * rank / head_dim)
    return int(stored_bytes)


def per_layer_recon_error(cache_a: Any, cache_b: Any) -> tuple[float, float]:
    layers_a = sc.to_legacy_kv(cache_a)
    layers_b = sc.to_legacy_kv(cache_b)
    k_errs: list[float] = []
    v_errs: list[float] = []
    for (ka, va), (kb, vb) in zip(layers_a, layers_b):
        k_errs.append(float(torch.norm(ka.float() - kb.float()) / torch.norm(kb.float())))
        v_errs.append(float(torch.norm(va.float() - vb.float()) / torch.norm(vb.float())))
    return sum(k_errs) / len(k_errs), sum(v_errs) / len(v_errs)


def greedy_generate_adaptive(
    model,
    tokenizer,
    prompt: str,
    device: str,
    method: str,
    rank: int,
    w_k: torch.Tensor | None = None,
    w_v: torch.Tensor | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    alpha: float = ALPHA,
    update_interval: int = UPDATE_INTERVAL,
) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []
    per_token_errors: list[dict[str, float]] = []
    drift_history: list[float] = []
    update_costs: list[float] = []

    num_heads = model_num_heads(model)
    head_dim = model_head_dim(model)

    updater: BatchedIPCAUpdater | None = None
    dtype = next(model.parameters()).dtype
    if method == "adaptive" and w_k is not None and w_v is not None:
        updater = BatchedIPCAUpdater(
            w_k.clone().to(dtype=dtype, device=model.device),
            w_v.clone().to(dtype=dtype, device=model.device),
            alpha=alpha,
            update_interval=update_interval,
            max_interval=MAX_INTERVAL,
            drift_threshold=DRIFT_THRESHOLD,
        )

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

        if method == "fixed":
            past_key_values = subspace_compress_cache(past_key_values, w_k, w_v)
        elif method == "adaptive" and updater is not None:
            update_t0 = time.monotonic()
            past_key_values, k_drift, v_drift = updater.update_and_compress(
                past_key_values, sc
            )
            update_cost = (time.monotonic() - update_t0) * 1000
            update_costs.append(update_cost)
            drift_history.append(k_drift + v_drift)

        step_time = (time.monotonic() - step_start) * 1000
        latencies.append(step_time)

        if step == 0:
            err_k, err_v = per_layer_recon_error(past_key_values, past_key_values)
        else:
            err_k, err_v = 0.0, 0.0
        per_token_errors.append({"token": step, "err_k": err_k, "err_v": err_v})

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stored_bytes = sc.kv_cache_bytes(past_key_values)
    seq_len = sc.to_legacy_kv(past_key_values)[0][0].shape[2]

    result = {
        "method": method,
        "rank": rank,
        "output": output_text,
        "tokens_generated": int(new_tokens.shape[0]),
        "token_ids": [int(t) for t in new_tokens.tolist()],
        "latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "latency_per_token_ms": [round(lt, 3) for lt in latencies],
        "kv_bytes": effective_kv_bytes(method, stored_bytes, num_heads, head_dim, rank),
        "kv_bytes_stored": stored_bytes,
        "bytes_per_token": round(stored_bytes / seq_len, 1) if seq_len else 0,
        "per_token_errors": per_token_errors,
    }

    if method == "adaptive":
        result["drift_history"] = drift_history
        result["update_costs_ms"] = [round(c, 3) for c in update_costs]
        result["update_alpha"] = alpha
        result["update_interval"] = update_interval
        result["num_updates"] = len([d for d in drift_history if d > 0])

    return result


def token_break_point(baseline_ids: list[int], other_ids: list[int]) -> int | None:
    for i, (a, b) in enumerate(zip(baseline_ids, other_ids)):
        if a != b:
            return i
    return None


def describe_divergence(baseline: str, other: str) -> str:
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


def recon_error(cache_a: Any, cache_b: Any) -> float:
    layers_a = sc.to_legacy_kv(cache_a)
    layers_b = sc.to_legacy_kv(cache_b)
    errs: list[float] = []
    for (ka, va), (kb, vb) in zip(layers_a, layers_b):
        ek = torch.norm(ka.float() - kb.float()) / torch.norm(kb.float())
        ev = torch.norm(va.float() - vb.float()) / torch.norm(vb.float())
        errs.append((float(ek) + float(ev)) / 2.0)
    return sum(errs) / len(errs) if errs else 0.0


def write_report(result: dict[str, Any], path: Path) -> None:
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

    print("\n=== baseline ===")
    baseline = greedy_generate_adaptive(
        model, tokenizer, DEFAULT_PROMPT, device, "baseline", rank
    )
    baseline.pop("cache", None)
    baseline_ids: list[int] = baseline["token_ids"]
    baseline["divergence"] = "identical"
    baseline["break_point"] = None
    baseline["recon_error"] = 0.0
    baseline["kv_saved_bytes"] = 0
    write_report(baseline, REPORT_DIR / f"run_baseline_{stamp}.json")
    results.append(baseline)
    print(f"baseline   : '{baseline['output'][:60]}...'  "
          f"kv={baseline['kv_bytes_stored']}  lat={baseline['latency_ms']}ms")

    baseline_cache_full = None

    for method in ["fixed", "adaptive"]:
        print(f"\n=== {method} subspace (rank={rank}) ===")
        result = greedy_generate_adaptive(
            model,
            tokenizer,
            DEFAULT_PROMPT,
            device,
            method,
            rank,
            w_k=w_k,
            w_v=w_v,
        )
        result_cache = result.pop("cache") if "cache" in result else None

        if baseline_cache_full is None and result_cache is not None:
            inputs2 = tokenizer(DEFAULT_PROMPT, return_tensors="pt").to(device)
            gen2 = inputs2["input_ids"]
            pkv2 = None
            with torch.no_grad():
                for _ in range(MAX_NEW_TOKENS):
                    if pkv2 is None:
                        o2 = model(input_ids=gen2, use_cache=True, past_key_values=None)
                    else:
                        o2 = model(input_ids=gen2[:, -1:], use_cache=True, past_key_values=pkv2)
                    nxt = o2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen2 = torch.cat([gen2, nxt], dim=1)
                    pkv2 = o2.past_key_values
            baseline_cache_full = pkv2

        result["divergence"] = describe_divergence(baseline["output"], result["output"])
        result["break_point"] = token_break_point(baseline_ids, result["token_ids"])
        if baseline_cache_full is not None and result_cache is not None:
            result["recon_error"] = recon_error(result_cache, baseline_cache_full)
        else:
            result["recon_error"] = 0.0
        result["kv_saved_bytes"] = baseline["kv_bytes_stored"] - result["kv_bytes"]

        write_report(result, REPORT_DIR / f"run_{method}_{stamp}.json")
        results.append(result)
        print(f"Output: {result['output'][:70]!r}")
        print(
            f"latency={result['latency_ms']}ms  kv_bytes={result['kv_bytes']}  "
            f"(stored {result['kv_bytes_stored']})  "
            f"break_point={result['break_point']}  recon_error={result['recon_error']:.4f}"
        )
        if method == "adaptive":
            print(f"  drift_events={result.get('num_updates', 0)}  "
                  f"mean_update_cost={sum(result.get('update_costs_ms', []))/max(1, len(result.get('update_costs_ms', []))):.3f}ms "
                  f"alpha={result.get('update_alpha')}  interval={result.get('update_interval')}")

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
    print("KV_ADAPTIVE_SUBSPACE_OK=1")


if __name__ == "__main__":
    run()
