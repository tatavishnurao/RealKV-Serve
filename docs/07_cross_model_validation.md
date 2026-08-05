# Cross-Model KV-Cache Validation (Milestone 6)

Validates whether the KV-cache properties observed on TinyLlama in
Milestones 2–5 **generalize across multiple real open-weight LLMs**. This is
purely a reproducible empirical-evidence milestone: no new compression methods,
no serving infrastructure, no batching, no distributed inference.

## Methodology

Every available model runs the **identical** workflow:

- Prompt: `"The capital of France is"` (identical plain text, no chat template)
- Decoding: greedy, `use_cache=True`, 32 new tokens
- Dtype: `float16` on CUDA (float32 fallback on CPU)
- Warm-up: one untimed decode pass to compile kernels before timing

Per model, three strategies are compared inside the same decode loop, reusing
the implementations from `run_kv_structured_compare.py` (Milestone 5):

| Strategy | Configs |
|---|---|
| baseline (full KV) | — |
| structured SVD low-rank | rank `head_dim/2`, `head_dim/4` |
| head pruning (zero dropped kv-heads) | keep 1/2, 1/4 of kv-heads |

Metrics collected per model:

- `kv_bytes` — effective compressed footprint and `kv_bytes_stored` (physical)
- `bytes/token` — stored KV bytes per sequence position at the final step
- `decode_latency_ms` — average per-token decode latency
- `prefill_latency_ms` — first forward pass latency
- `gpu_memory_mb` — peak PyTorch GPU allocation
- `layers`, `kv_heads`, `head_dim`, `parameters` — architecture facts
- `svd_error` / `head_prune_error` — mean relative K/V reconstruction error
- `effective_rank`, `rank90` — singular-value energy structure of the cache
- `break_point` — first generated token that diverges from that model's baseline

Models that cannot be downloaded, are gated, or do not fit in VRAM are skipped
**automatically** with a recorded `skipped_reason`. Reports: one JSON per model
plus one summary JSON under `reports/cross_model/`.

## Hardware / Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GiB VRAM) |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.14.1 |
| Runner | `uv run python scripts/run_cross_model_compare.py` |

## Supported vs Skipped Models

| Model | Status | Reason |
|---|---|---|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ✅ ran | — |
| `Qwen/Qwen2.5-1.5B-Instruct` | ✅ ran | — |
| `Qwen/Qwen2.5-3B-Instruct` | ✅ ran | — |
| `microsoft/Phi-3-mini-4k-instruct` | ⛔ skipped | CUDA OOM — 3.8B params in fp16 ≈ 7.6 GiB of weights alone exceeds the 8 GiB card (with ambient GPU processes) |
| `google/gemma-2-2b-it` | ⛔ skipped | HF gated repo — requires an authenticated token to download |

## Summary Table (run `20260805_171641`)

| Model | Parameters | Layers | KV Heads | Head Dim | KV Bytes | Bytes/Token | SVD Rank | SVD Error | Head Prune Error | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | 1.10B | 22 | 4 | 64 | 0.83M | 22.5K | 16 | 0.194 | 0.866 | 12.7 |
| Qwen2.5-1.5B-Instruct | 1.54B | 28 | 2 | 128 | 1.03M | 28.7K | 32 | 0.037 | 0.709 | 15.1 |
| Qwen2.5-3B-Instruct | 3.09B | 36 | 2 | 128 | 1.33M | 36.9K | 32 | 0.041 | 0.712 | 28.7 |

(SVD Rank = `head_dim/4`; SVD Error = mean relative K/V reconstruction error at
that rank; Head Prune Error = mean relative K/V error keeping 1/4 of kv-heads.)

## Detected Trends (automated, data-driven)

- **KV bytes grow linearly with sequence length across models.**
  bytes/token coefficient of variation per model = `0.0` for all three.
  bytes/token is exactly `2 × layers × kv_heads × head_dim × element_size`
  (e.g. TinyLlama `2·22·4·64·2 = 22,528`; Qwen `2·28·2·128·2 = 28,672`), i.e.
  fully predictable from architecture alone.
- **SVD energy structure is similar across models.** The rank retaining 90% of
  KV Frobenius energy is only ~6–11% of `head_dim`
  (range 0.059–0.111, CV=0.304 → "similar"), and the effective (participation)
  rank is ~2.1–2.2 for every model.
- **SVD preserves output exactly on all models.** Both SVD ranks produced the
  baseline 32-token output bit-for-bit on 3/3 models (`break_point = None`).
- **SVD far outperforms head pruning on reconstruction.** SVD error is below
  head-prune error on 3/3 models (SVD: 0.037–0.194; prune: 0.709–0.866).
- **Latency and KV bytes scale with model size.** r(params, KV bytes) = 0.98,
  r(params, decode latency) = 1.00, r(params, prefill) = 0.74.

## Observed Similarities Across Models

1. **Near-lossless low-rank KV.** Despite different families, head dims (64 vs
   128), kv-head counts (4 vs 2), and layer counts (22–36), the KV cache is
   extremely low-rank: effective rank ≈ 2 and 90% energy in ~7–8 components.
   Rank-`head_dim/4` SVD changed nothing about the decoded output.
2. **Perfectly linear, architecture-determined KV growth.** No model deviates
   from a constant bytes/token.
3. **Head pruning is the consistent baseline.** Keeping 1/4 of kv-heads costs
   ~0.71–0.87 relative error everywhere — always worse than SVD, always
   predictably bad.
4. **Decode latency tracks parameter count.** r = 1.00 across the three sizes.

## Observed Differences Across Models

1. **Absolute SVD reconstruction quality.** TinyLlama's rank-16 SVD error
   (0.194) is ~5× higher than the Qwen models' rank-32 error (0.037–0.041) —
   larger head dims give SVD much more room to be lossless. Yet output was
   identical on all three, so the larger error is still below the argmax
   sensitivity threshold for this task.
2. **KV footprint per token.** Larger kv-head × head-dim products push
   bytes/token from 22.5K (TinyLlama) to 36.9K (Qwen-3B) — a consequence of
   architecture, not of anything compressible.
3. **Prefill correlation is weaker** (r = 0.74) than decode latency (r = 1.00),
   consistent with prefill being dominated by prompt-shape/compile effects.

## What This Means for the KV Compression Question

The Milestone-5 result — *"KV is compressible without breaking attention"* —
**generalizes**: on every model tested, SVD at `head_dim/4` preserved output
exactly while cutting the effective KV footprint 4×, and the KV energy
distribution is similarly low-rank across all three families. Head pruning
remains a simple, consistent, but lossier baseline, and token-pooling-style
temporal collapse was already shown (M5) to destroy coherence. These are
empirical facts on 3 models / 8 GiB VRAM; a broader sample (including
Phi-3-class and gated models) requires more VRAM and HF auth.

## Running

```bash
bash scripts/run_cross_model_compare.sh
# or
python scripts/run_cross_model_compare.py
```

Success marker:

```text
CROSS_MODEL_COMPARE_OK=1
```

Reports:

```text
reports/cross_model/<ModelName>_<timestamp>.json   # one per model (or skip record)
reports/cross_model/summary_<timestamp>.json       # table + trends + all models
```

## Relation to Earlier Milestones

- M2 (`docs/02_real_kv_trace.md`): healthy linear KV growth on one model.
- M4 (`docs/04_kv_latent_compare.md`): random projections catastrophically break output.
- M5 (`docs/05_kv_structured_compare.md`): SVD lossless, head pruning graceful,
  token pooling incoherent — on TinyLlama.
- M6 (this doc): those findings generalize to three open-weight LLMs.
