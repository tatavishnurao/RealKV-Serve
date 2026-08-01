"""KV-cache stress experiments: controlled perturbations during decoding.

Measures output divergence, KV size, and latency under truncation,
zeroing (corruption), and noise injection on a real causal LM.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
REPORT_DIR = Path("reports/kv_stress")


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
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        return [(k, v) for k, v in legacy]
    return [(layer[0], layer[1]) for layer in past_key_values]


def from_legacy_kv(
    layers: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Convert list of (k, v) pairs to a tuple past_key_values."""
    return tuple(layers)


def truncate_kv(
    past_key_values: Any,
    n: int | None,
) -> Any:
    """Keep only the last N positions of each layer's K/V tensors.

    n=None means no truncation (full cache). Simulates reduced context
    windows such as aggressive sliding-window attention.
    """
    if past_key_values is None or n is None:
        return past_key_values
    layers = to_legacy_kv(past_key_values)
    truncated: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in layers:
        seq_len = k.shape[2]
        keep = min(n, seq_len)
        if keep <= 0:
            keep = seq_len
        truncated.append((k[:, :, -keep:, :].contiguous(), v[:, :, -keep:, :].contiguous()))
    return from_legacy_kv(truncated)


def resolve_truncate_n(mode: str | int | None, current_seq_len: int) -> int | None:
    """Map truncation mode to a concrete keep-N value."""
    if mode is None or mode == "full":
        return None
    if mode == "half":
        return max(1, current_seq_len // 2)
    return int(mode)


def zero_kv(past_key_values: Any) -> Any:
    """Zero all key and value tensors in the cache (hard corruption).

    Simulates catastrophic cache wipe / memory corruption at a fixed step.
    Operates in-place on tensor storage when possible, then returns a
    legacy tuple so subsequent decode steps see the wiped state.
    """
    if past_key_values is None:
        return None
    layers = to_legacy_kv(past_key_values)
    wiped: list[tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in layers:
        k = k.clone()
        v = v.clone()
        k.zero_()
        v.zero_()
        wiped.append((k, v))
    return from_legacy_kv(wiped)


def load_model(device: str):
    """Load TinyLlama causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        use_cache=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    experiment: str = "baseline",
    param: str | int | float = "full",
    truncate_mode: str | int | None = None,
    zero_at_step: int | None = None,
) -> dict[str, Any]:
    """Greedy decode with optional KV truncation and/or one-shot zeroing."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []

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

        # Apply KV truncation after each step (reduced-context simulation).
        if truncate_mode is not None and truncate_mode != "full":
            layers = to_legacy_kv(past_key_values)
            if layers:
                cur_len = layers[0][0].shape[2]
                n = resolve_truncate_n(truncate_mode, cur_len)
                past_key_values = truncate_kv(past_key_values, n)

        # One-shot hard corruption at a fixed decode step.
        if zero_at_step is not None and step == zero_at_step:
            past_key_values = zero_kv(past_key_values)

        latencies.append((time.monotonic() - step_start) * 1000)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "experiment": experiment,
        "param": param,
        "output": output_text,
        "tokens_generated": int(new_tokens.shape[0]),
        "latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "kv_bytes": kv_cache_bytes(past_key_values),
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


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    model, tokenizer = load_model(device)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results: list[dict[str, Any]] = []

    print("\n=== Truncation: full (baseline) ===")
    baseline = greedy_generate(
        model,
        tokenizer,
        DEFAULT_PROMPT,
        device,
        experiment="truncation",
        param="full",
        truncate_mode="full",
    )
    print(f"Output: {baseline['output']!r}")
    print(f"Avg latency: {baseline['latency_ms']} ms  KV bytes: {baseline['kv_bytes']}")
    write_report(baseline, REPORT_DIR / f"truncation_full_{stamp}.json")
    results.append(baseline)
    baseline_text = baseline["output"]

    for mode, label in [("half", "half"), (8, "last8")]:
        print(f"\n=== Truncation: {label} ===")
        result = greedy_generate(
            model,
            tokenizer,
            DEFAULT_PROMPT,
            device,
            experiment="truncation",
            param=mode if mode != 8 else 8,
            truncate_mode=mode,
        )
        result["divergence"] = describe_divergence(baseline_text, result["output"])
        print(f"Output: {result['output']!r}")
        print(f"Avg latency: {result['latency_ms']} ms  KV bytes: {result['kv_bytes']}")
        print(f"Divergence: {result['divergence']}")
        write_report(result, REPORT_DIR / f"truncation_{label}_{stamp}.json")
        results.append(result)

    zero_step = 10
    print(f"\n=== Zeroing corruption at step {zero_step} ===")
    zero_result = greedy_generate(
        model,
        tokenizer,
        DEFAULT_PROMPT,
        device,
        experiment="zeroing",
        param=zero_step,
        zero_at_step=zero_step,
    )
    zero_result["divergence"] = describe_divergence(baseline_text, zero_result["output"])
    print(f"Output: {zero_result['output']!r}")
    print(f"Avg latency: {zero_result['latency_ms']} ms  KV bytes: {zero_result['kv_bytes']}")
    print(f"Divergence: {zero_result['divergence']}")
    write_report(zero_result, REPORT_DIR / f"zeroing_step{zero_step}_{stamp}.json")
    results.append(zero_result)

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
    print("KV_STRESS_RUN_OK=1")


if __name__ == "__main__":
    run()
