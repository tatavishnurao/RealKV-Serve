"""Build a reusable low-rank KV subspace basis from a real KV trace.

Runs a KV trace on a model (prefill + greedy decode), computes the top-k right
singular vectors of each layer's K and V matrices (kv-heads stacked along the
sequence dimension), and saves:

  reports/kv_subspace/<model>/basis.pt        (W_k, W_v orthonormal per layer)
  reports/kv_subspace/<model>/basis_meta.json (rank, explained variance, config)

The basis is a fixed, learned-from-data subspace. Decoding reuses it by
projecting KV in and reconstructing before attention, without recomputing SVD.
This turns the SVD analysis of Milestones 5-6 into a reusable component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_kv_structured_compare as sc

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
TRACE_TOKENS = 64
RANK_DIVISOR = 4
REPORT_DIR = Path("reports/kv_subspace")


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


def trace_cache(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_tokens: int,
) -> Any:
    """Prefill + greedy decode; returns the final DynamicCache (KV sample)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = inputs["input_ids"]
    past_key_values = None
    with torch.no_grad():
        out = model(input_ids=generated, use_cache=True, past_key_values=None)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        past_key_values = out.past_key_values
        for _ in range(max_tokens - 1):
            out = model(
                input_ids=generated[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            past_key_values = out.past_key_values
    return past_key_values


def build_basis(
    cache: Any,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """Compute per-layer orthonormal KV bases via truncated SVD.

    Each layer's K and V matrices are reshaped to [kv_heads * seq, head_dim]
    and truncated-SVD'd. The top-`rank` right singular vectors form an
    orthonormal [head_dim, rank] basis per layer per tensor type.

    Returns (W_k, W_v, explained_var_k, explained_var_v, projection_error).
    """
    layers = sc.to_legacy_kv(cache)
    dim = layers[0][0].shape[-1]
    w_k = torch.zeros(len(layers), dim, rank)
    w_v = torch.zeros(len(layers), dim, rank)
    ev_k: list[float] = []
    ev_v: list[float] = []
    proj_errs: list[float] = []
    for i, (k, v) in enumerate(layers):
        k2 = k.reshape(-1, dim).float()
        v2 = v.reshape(-1, dim).float()
        _, s_k, vh_k = torch.linalg.svd(k2, full_matrices=False)
        _, s_v, vh_v = torch.linalg.svd(v2, full_matrices=False)
        r = min(rank, vh_k.shape[0])
        w_k[i, :, :r] = vh_k[:r].t()
        w_v[i, :, :r] = vh_v[:r].t()
        ev_k.append(float((s_k[:r] ** 2).sum() / (s_k**2).sum()))
        ev_v.append(float((s_v[:r] ** 2).sum() / (s_v**2).sum()))
        w = w_k[i].to(device=k.device, dtype=k.dtype)
        k_recon = (k @ w) @ w.t()
        proj_errs.append(float(torch.norm(k_recon.float() - k.float()) / torch.norm(k.float())))
    return (
        w_k,
        w_v,
        sum(ev_k) / len(ev_k),
        sum(ev_v) / len(ev_v),
        sum(proj_errs) / len(proj_errs),
    )


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write one structured JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    model, tokenizer = load_model(device)
    sc.warmup(model, tokenizer, device)

    num_heads = int(
        getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    )
    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    head_dim = int(head_dim)
    rank = max(1, head_dim // RANK_DIVISOR)

    print(f"Tracing {TRACE_TOKENS} tokens to sample the KV distribution ...")
    cache = trace_cache(model, tokenizer, DEFAULT_PROMPT, device, TRACE_TOKENS)
    print(f"Traced cache: seq_len={sc.to_legacy_kv(cache)[0][0].shape[2]}, "
          f"kv_bytes={sc.kv_cache_bytes(cache)}")

    w_k, w_v, ev_k, ev_v, proj_err = build_basis(cache, rank)
    print(f"Basis rank: {rank}  explained variance: K={ev_k:.4f} V={ev_v:.4f}  "
          f"one-shot projection error: {proj_err:.4f}")

    slug = MODEL_NAME.split("/")[-1]
    out_dir = REPORT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    basis_path = out_dir / "basis.pt"

    torch.save(
        {
            "model": MODEL_NAME,
            "W_k": w_k.cpu(),
            "W_v": w_v.cpu(),
            "rank": rank,
            "head_dim": head_dim,
            "num_layers": w_k.shape[0],
        },
        basis_path,
    )

    meta = {
        "model": MODEL_NAME,
        "prompt": DEFAULT_PROMPT,
        "trace_tokens": TRACE_TOKENS,
        "rank": rank,
        "head_dim": head_dim,
        "num_layers": int(model.config.num_hidden_layers),
        "kv_heads": num_heads,
        "explained_variance_k": round(ev_k, 4),
        "explained_variance_v": round(ev_v, 4),
        "projection_error": round(proj_err, 4),
        "basis_path": str(basis_path),
    }
    write_json(meta, out_dir / "basis_meta.json")

    print(f"\nBasis written to {basis_path}")
    print(f"Metadata written to {out_dir / 'basis_meta.json'}")
    print("KV_SUBSPACE_BUILT_OK=1")


if __name__ == "__main__":
    run()
