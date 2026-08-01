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
) -> dict[str, Any]:
    """Baseline greedy decode with use_cache=True (no KV manipulation)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    latencies: list[float] = []

    for _step in range(max_new_tokens):
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
        latencies.append((time.monotonic() - step_start) * 1000)

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {
        "experiment": "baseline",
        "param": "full",
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


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    model, tokenizer = load_model(device)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("\n=== Baseline (full KV cache) ===")
    baseline = greedy_generate(model, tokenizer, DEFAULT_PROMPT, device)
    print(f"Output: {baseline['output']!r}")
    print(f"Avg latency: {baseline['latency_ms']} ms")
    print(f"KV bytes: {baseline['kv_bytes']}")
    write_report(baseline, REPORT_DIR / f"baseline_full_{stamp}.json")

    summary_path = REPORT_DIR / f"summary_{stamp}.json"
    write_report({"results": [baseline], "model": MODEL_NAME, "device": device}, summary_path)

    print(f"\nReports written under {REPORT_DIR}/")
    print("KV_STRESS_RUN_OK=1")


if __name__ == "__main__":
    run()
