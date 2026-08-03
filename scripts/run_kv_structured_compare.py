"""Structure-preserving KV compression vs baseline full KV during real decoding.

Evaluates compression methods that try to keep attention structure instead of
blindly projecting each KV entry through a random bottleneck (Milestone 4):

- head_prune: drop (zero) a subset of attention heads' K/V entirely
- svd:        low-rank SVD approximation of each layer's K/V matrices
- token_pool: temporally merge groups of sequence positions (mean pooling)

Each method is applied to the live cache after every decode step and compared
against an unmodified baseline. All transforms are shape-preserving (or keep
the cache contract intact) so the same greedy decode loop runs end-to-end.

No CUDA kernels, no model training: pure PyTorch tensor ops on a real LM.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
REPORT_DIR = Path("reports/kv_structured_compare")

# (method, compression_ratio) pairs exercised by run().
EXPERIMENTS = [
    ("head_prune", 2),
    ("head_prune", 4),
    ("svd", 2),
    ("svd", 4),
    ("token_pool", 2),
    ("token_pool", 4),
]


def kv_cache_bytes(past_key_values: Any) -> int:
    """Return total bytes consumed by all KV-cache tensors."""
    if past_key_values is None:
        return 0
    layers = to_legacy_kv(past_key_values)
    total = 0
    for k, v in layers:
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return total


def to_legacy_kv(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Normalize past_key_values to a list of (key, value) tensor pairs."""
    if past_key_values is None:
        return []
    # transformers 5+ DynamicCache
    if hasattr(past_key_values, "layers"):
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                out.append((keys, values))
        if out:
            return out
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [(k, v) for k, v in legacy]
    return [(layer[0], layer[1]) for layer in past_key_values]


def from_legacy_kv(layers: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Rebuild a DynamicCache (transformers 5+) from (k, v) pairs."""
    return DynamicCache(ddp_cache_data=layers)


def head_prune_cache(
    past_key_values: Any,
    keep_heads: int,
    num_heads: int,
) -> Any:
    """Zero K/V of pruned heads (shape-preserving head pruning).

    Pruned heads see all-zero keys and values, so they contribute nothing to
    the attention output -- equivalent to removing them while keeping tensor
    shapes valid for the next forward pass.
    """
    layers = to_legacy_kv(past_key_values)
    device = layers[0][0].device
    mask = torch.zeros(num_heads, dtype=torch.bool, device=device)
    mask[: min(keep_heads, num_heads)] = True
    mask = mask.view(1, -1, 1, 1)
    pruned: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in layers:
        k = k.clone().masked_fill(~mask, 0)
        v = v.clone().masked_fill(~mask, 0)
        pruned.append((k, v))
    return from_legacy_kv(pruned)


def svd_approx(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Best rank-`rank` approximation of [*, seq, dim] via truncated SVD.

    Computed in float32 because CUDA SVD is not implemented for float16.
    """
    orig_dtype = x.dtype
    u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    r = min(rank, u.shape[-1])
    approx = (u[..., :r] @ torch.diag_embed(s[..., :r])) @ vh[..., :r, :]
    return approx.to(orig_dtype)


def svd_compress_cache(past_key_values: Any, rank: int) -> Any:
    """Replace every layer's K/V with its truncated-SVD reconstruction."""
    layers = to_legacy_kv(past_key_values)
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in layers:
        rebuilt.append((svd_approx(k, rank).contiguous(), svd_approx(v, rank).contiguous()))
    return from_legacy_kv(rebuilt)


def pool_seq(x: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Mean-pool groups of `pool_size` tokens along the sequence dimension."""
    seq = x.shape[2]
    n = seq // pool_size
    if n == 0:
        return x
    main = x[:, :, : n * pool_size, :]
    pooled = main.unfold(2, pool_size, pool_size).mean(dim=-1)
    if n * pool_size < seq:
        pooled = torch.cat([pooled, x[:, :, n * pool_size :, :]], dim=2)
    return pooled.contiguous()


def pool_tokens_cache(past_key_values: Any, pool_size: int) -> Any:
    """Merge groups of sequence positions into single pooled entries."""
    layers = to_legacy_kv(past_key_values)
    pooled: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in layers:
        pooled.append((pool_seq(k, pool_size), pool_seq(v, pool_size)))
    return from_legacy_kv(pooled)


def resolve_keep_heads(num_heads: int, compression_ratio: int) -> int:
    """Map a compression ratio to a kept-head count (num_heads // ratio)."""
    return max(1, num_heads // max(1, compression_ratio))


def resolve_rank(head_dim: int, compression_ratio: int) -> int:
    """Map a compression ratio to an SVD rank (head_dim // ratio)."""
    return max(1, head_dim // max(1, compression_ratio))


def effective_kv_bytes(
    method: str,
    compression_ratio: int,
    num_heads: int,
    head_dim: int,
    stored_bytes: int,
) -> int:
    """Effective on-disk footprint a real compressed-KV system would pay.

    - head_prune: keep `num_heads // ratio` heads' KV
    - svd:        store only the rank-`head_dim // ratio` subspace fraction
    - token_pool: the cache physically holds pooled (shorter-sequence) tensors
    """
    if method == "head_prune":
        keep = resolve_keep_heads(num_heads, compression_ratio)
        return int(stored_bytes * keep / num_heads)
    if method == "svd":
        rank = resolve_rank(head_dim, compression_ratio)
        return int(stored_bytes * rank / head_dim)
    if method == "token_pool":
        return int(stored_bytes)
    return int(stored_bytes)


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


def warmup(model, tokenizer, device: str) -> None:
    """Run one untimed decode pass to compile kernels and warm the GPU."""
    inputs = tokenizer(DEFAULT_PROMPT, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            out = model(
                input_ids=generated[:, -1:],
                use_cache=True,
                past_key_values=None,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)


def apply_compression(
    past_key_values: Any,
    method: str,
    compression_ratio: int,
    num_heads: int,
    head_dim: int,
) -> Any:
    """Apply one structure-preserving KV transform to the live cache."""
    if method == "head_prune":
        return head_prune_cache(
            past_key_values,
            resolve_keep_heads(num_heads, compression_ratio),
            num_heads,
        )
    if method == "svd":
        return svd_compress_cache(
            past_key_values,
            resolve_rank(head_dim, compression_ratio),
        )
    if method == "token_pool":
        return pool_tokens_cache(past_key_values, compression_ratio)
    return past_key_values


def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    method: str = "baseline",
    compression_ratio: int = 1,
) -> dict[str, Any]:
    """Greedy decode with full KV (baseline) or one compressed-KV strategy."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []

    num_heads = int(
        getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    )
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // num_heads
    head_dim = int(head_dim)

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

        if method != "baseline":
            past_key_values = apply_compression(
                past_key_values,
                method,
                compression_ratio,
                num_heads,
                head_dim,
            )

        latencies.append((time.monotonic() - step_start) * 1000)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    stored_bytes = kv_cache_bytes(past_key_values)
    return {
        "method": method,
        "compression_ratio": compression_ratio,
        "output": output_text,
        "tokens_generated": int(new_tokens.shape[0]),
        "token_ids": [int(t) for t in new_tokens.tolist()],
        "latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "kv_bytes": effective_kv_bytes(
            method,
            compression_ratio,
            num_heads,
            head_dim,
            stored_bytes,
        ),
        "kv_bytes_stored": stored_bytes,
        "divergence": None,
        "break_point": None,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    """Write one experiment result as structured JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


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


def common_prefix_chars(a: str, b: str) -> int:
    """Shared character prefix length between two decoded texts."""
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return prefix


def token_break_point(baseline_ids: list[int], other_ids: list[int]) -> int | None:
    """Index of the first generated token that differs from baseline."""
    for i, (a, b) in enumerate(zip(baseline_ids, other_ids)):
        if a != b:
            return i
    return None


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    model, tokenizer = load_model(device)
    warmup(model, tokenizer, device)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results: list[dict[str, Any]] = []

    print("\n=== Baseline: full KV ===")
    baseline = greedy_generate(model, tokenizer, DEFAULT_PROMPT, device)
    print(f"Output: {baseline['output']!r}")
    print(f"Avg latency: {baseline['latency_ms']} ms  KV bytes: {baseline['kv_bytes']}")
    write_report(baseline, REPORT_DIR / f"baseline_{stamp}.json")
    results.append(baseline)
    baseline_text = baseline["output"]
    baseline_bytes = baseline["kv_bytes"]
    baseline_ids: list[int] = baseline["token_ids"]

    for method, ratio in EXPERIMENTS:
        print(f"\n=== {method}: compression_ratio={ratio} ===")
        result = greedy_generate(
            model,
            tokenizer,
            DEFAULT_PROMPT,
            device,
            method=method,
            compression_ratio=ratio,
        )
        result["divergence"] = describe_divergence(baseline_text, result["output"])
        result["common_prefix_chars"] = common_prefix_chars(
            baseline_text,
            result["output"],
        )
        result["break_point"] = token_break_point(baseline_ids, result["token_ids"])
        result["kv_saved_bytes"] = baseline_bytes - result["kv_bytes"]
        print(f"Output: {result['output']!r}")
        print(
            f"Avg latency: {result['latency_ms']} ms  KV bytes: {result['kv_bytes']}  "
            f"(stored {result['kv_bytes_stored']})"
        )
        print(f"Break point: token {result['break_point']}  {result['divergence']}")
        write_report(result, REPORT_DIR / f"{method}_ratio{ratio}_{stamp}.json")
        results.append(result)

    summary_path = REPORT_DIR / f"summary_{stamp}.json"
    write_report(
        {
            "model": MODEL_NAME,
            "device": device,
            "prompt": DEFAULT_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "results": results,
        },
        summary_path,
    )

    print(f"\nReports written under {REPORT_DIR}/")
    print("KV_STRUCTURED_COMPARE_OK=1")


if __name__ == "__main__":
    run()
