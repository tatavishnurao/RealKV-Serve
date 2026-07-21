"""One-request deterministic generation entry point for a TRT-LLM container."""

from __future__ import annotations
import argparse
import json
import os
import time
import uuid
from .metrics import summarize_timestamps


def run(args: argparse.Namespace) -> dict:
    try:
        import torch
        from tensorrt_llm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Run this entry point inside the pinned TensorRT-LLM container") from exc
    torch.manual_seed(args.seed)
    request_id, start = f"req-{uuid.uuid4().hex[:12]}", time.monotonic_ns()
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)
    llm = LLM(model=args.model, backend="pytorch", dtype="float16")
    result = llm.generate([args.prompt], params)
    end = time.monotonic_ns()
    output = result[0].outputs[0]
    token_ids = list(output.token_ids)
    text = output.text
    if not token_ids or not text:
        raise RuntimeError("generation validation failed")
    payload = {
        "request_id": request_id,
        "input_text": args.prompt,
        "input_token_ids": None,
        "input_token_count": None,
        "generated_token_ids": token_ids,
        "generated_text": text,
        "output_token_count": len(token_ids),
        "generation_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_ms": (end - start) / 1e6,
        "runtime_configuration": vars(args),
        "model_revision": os.environ.get("REALKV_MODEL_REVISION", "unknown"),
        "timing": summarize_timestamps(start, [], end),
    }
    print("REAL_MODEL_GENERATION_OK=1")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run(args)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
