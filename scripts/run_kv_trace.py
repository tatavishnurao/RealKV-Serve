"""Causal LM inference with explicit KV-cache lifecycle tracing.

Runs greedy decoding on a real HuggingFace model and logs KV-cache
metadata per decode step to a JSONL trace file.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def kv_cache_bytes(past_key_values) -> int:
    """Return total bytes consumed by all KV-cache tensors."""
    total = 0
    for layer in past_key_values:
        for tensor in layer:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def kv_shapes(past_key_values) -> list[list[int]]:
    """Return list of [k_shape, v_shape] per layer."""
    shapes = []
    for layer in past_key_values:
        layer_shapes = []
        for t in layer:
            if t is not None:
                layer_shapes.append(list(t.shape))
        shapes.append(layer_shapes)
    return shapes


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_cache=True,
    ).to(device)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    max_new_tokens = 32
    generated = inputs["input_ids"]
    past_key_values = None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports/kv_trace", exist_ok=True)
    trace_path = f"reports/kv_trace/run_{timestamp}.jsonl"

    with open(trace_path, "w", encoding="utf-8") as f:
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

            step_end = time.monotonic()
            token_latency_ms = (step_end - step_start) * 1000

            seq_len = generated.shape[1]
            kv_bytes = kv_cache_bytes(past_key_values)
            gpu_alloc = torch.cuda.memory_allocated() / 1024**2
            gpu_reserved = torch.cuda.memory_reserved() / 1024**2

            record = {
                "step": step,
                "seq_len": seq_len,
                "num_layers": len(past_key_values),
                "kv_shapes_per_layer": kv_shapes(past_key_values),
                "total_kv_bytes": kv_bytes,
                "total_kv_mib": kv_bytes / 1024**2,
                "gpu_allocated_mb": round(gpu_alloc, 2),
                "gpu_reserved_mb": round(gpu_reserved, 2),
                "token_latency_ms": round(token_latency_ms, 3),
                "device": device,
                "model": model_name,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            decoded = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(
                f"STEP={step:2d}  SEQ_LEN={seq_len:3d}  "
                f"KV_BYTES={kv_bytes:>10d}  "
                f"GPU_ALLOC={gpu_alloc:7.2f}MiB  "
                f"LATENCY={token_latency_ms:7.3f}ms  "
                f"TOKEN={decoded!r}"
            )

    print(f"\nKV trace written to {trace_path}")
    print(f"Total KV bytes at final step: {kv_cache_bytes(past_key_values)}")
    print(f"Peak GPU allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")
    print("KV_TRACE_RUN_OK=1")


if __name__ == "__main__":
    run()
