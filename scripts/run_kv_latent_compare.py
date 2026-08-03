"""Simulated latent KV compression vs baseline full KV during real decoding.

Runs greedy decoding with `use_cache=True` on a real causal LM and compares:

- baseline: store full K/V in the cache each step
- latent:   project each layer's K/V through a linear bottleneck
            (W_down: [head_dim -> latent_dim], W_up: [latent_dim -> head_dim])
            and store the reconstructed K/V in the cache.

The projection is a seeded orthonormal linear map (W_up = W_down^T), so the
bottleneck behaves like a rank-`latent_dim` PCA-style approximation of each KV
entry. Because an orthogonal projection is idempotent, already-compressed
positions are not re-damaged on later steps; only each newly appended KV entry
carries compression loss.

No CUDA kernels are used: compression is a plain `torch.matmul`. This is a
*simulation* of a latent KV system, not an integration of one.
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
COMPRESSION_RATIOS = [2, 4, 8]
REPORT_DIR = Path("reports/kv_latent_compare")
PROJ_SEED = 0


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


def resolve_head_dim(config: Any) -> int:
    """Determine per-head K/V feature dimension from a model config."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // getattr(config, "num_attention_heads", 32)
    return int(head_dim)


def resolve_latent_dim(head_dim: int, compression_ratio: int) -> int:
    """Map a compression ratio to a latent dimension (head_dim // ratio)."""
    return max(1, head_dim // max(1, compression_ratio))


def build_projection(
    head_dim: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = PROJ_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (W_down, W_up) with W_down an orthonormal [head_dim, latent_dim] map.

    W_up is the transpose of W_down, so W_down @ W_up is an orthogonal
    projection onto the column space of W_down (rank-`latent_dim` bottleneck).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    w_down = torch.randn(head_dim, latent_dim, device=device, generator=gen).float()
    w_down, _ = torch.linalg.qr(w_down)
    w_down = w_down.to(dtype)
    w_up = w_down.transpose(-1, -2)
    return w_down, w_up


def compress_latent(
    past_key_values: Any,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
) -> tuple[Any, int]:
    """Lossy-compress every K/V via a latent bottleneck and rebuild the cache.

    Returns (rebuilt DynamicCache holding reconstructed K/V, latent_bytes)
    where latent_bytes is the effective on-disk footprint of the compressed
    representation (what a real latent KV system would store).
    """
    layers = to_legacy_kv(past_key_values)
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
    latent_bytes = 0
    for k, v in layers:
        latent_k = k @ w_down
        latent_v = v @ w_down
        latent_bytes += latent_k.numel() * latent_k.element_size()
        latent_bytes += latent_v.numel() * latent_v.element_size()
        k_recon = latent_k @ w_up
        v_recon = latent_v @ w_up
        rebuilt.append((k_recon.contiguous(), v_recon.contiguous()))
    return from_legacy_kv(rebuilt), latent_bytes


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
    """Run one untimed decode pass to compile kernels and warm the GPU.

    Decodes with the real prompt and max_new_tokens so every sequence-length
    shape the measured runs will hit is already compiled.
    """
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


def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    mode: str = "baseline",
    compression_ratio: int = 1,
) -> dict[str, Any]:
    """Greedy decode with either full KV (baseline) or simulated latent KV."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []
    latent_bytes = 0
    kv_bytes = 0

    w_down: torch.Tensor | None = None
    w_up: torch.Tensor | None = None
    if mode == "latent":
        head_dim = resolve_head_dim(model.config)
        latent_dim = resolve_latent_dim(head_dim, compression_ratio)
        w_down, w_up = build_projection(
            head_dim,
            latent_dim,
            generated.device,
            next(model.parameters()).dtype,
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

        if mode == "latent":
            past_key_values, latent_bytes = compress_latent(
                past_key_values,
                w_down,
                w_up,
            )

        latencies.append((time.monotonic() - step_start) * 1000)

    if mode == "latent":
        kv_bytes = latent_bytes
    else:
        kv_bytes = kv_cache_bytes(past_key_values)
    kv_bytes_stored = kv_cache_bytes(past_key_values)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "mode": mode,
        "compression_ratio": compression_ratio,
        "output": output_text,
        "tokens_generated": int(new_tokens.shape[0]),
        "latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "kv_bytes": kv_bytes,
        "kv_bytes_stored": kv_bytes_stored,
        "divergence": None,
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


def common_prefix_tokens(baseline: str, other: str) -> int:
    """Count of shared prefix characters between two decoded texts."""
    prefix = 0
    for a, b in zip(baseline, other):
        if a != b:
            break
        prefix += 1
    return prefix


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
    baseline = greedy_generate(
        model,
        tokenizer,
        DEFAULT_PROMPT,
        device,
        mode="baseline",
        compression_ratio=1,
    )
    print(f"Output: {baseline['output']!r}")
    print(
        f"Avg latency: {baseline['latency_ms']} ms  "
        f"KV bytes: {baseline['kv_bytes']}"
    )
    write_report(baseline, REPORT_DIR / f"baseline_{stamp}.json")
    results.append(baseline)
    baseline_text = baseline["output"]
    baseline_bytes = baseline["kv_bytes"]

    for ratio in COMPRESSION_RATIOS:
        print(f"\n=== Latent KV: compression_ratio={ratio} ===")
        result = greedy_generate(
            model,
            tokenizer,
            DEFAULT_PROMPT,
            device,
            mode="latent",
            compression_ratio=ratio,
        )
        result["divergence"] = describe_divergence(baseline_text, result["output"])
        result["common_prefix_chars"] = common_prefix_tokens(
            baseline_text,
            result["output"],
        )
        result["kv_saved_bytes"] = baseline_bytes - result["kv_bytes"]
        print(f"Output: {result['output']!r}")
        print(
            f"Avg latency: {result['latency_ms']} ms  "
            f"KV bytes: {result['kv_bytes']}  "
            f"(stored {result['kv_bytes_stored']})"
        )
        print(f"Divergence: {result['divergence']}")
        write_report(result, REPORT_DIR / f"latent_ratio{ratio}_{stamp}.json")
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
    print("KV_LATENT_COMPARE_OK=1")


if __name__ == "__main__":
    run()
