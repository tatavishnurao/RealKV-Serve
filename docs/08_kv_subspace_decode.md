# Reusable Low-Rank KV Subspace Decoding (Milestone 7)

Milestones 5–6 showed that transformer KV caches are very low-rank and that
per-step truncated SVD preserves output exactly — but recomputing SVD every
step is ~12× slower than baseline. This milestone turns that SVD **analysis**
into a **reusable system component**: learn a KV subspace basis **once**,
cache it on disk, and reuse it during decoding at ~baseline latency.

No batching, no distributed inference, no serving infrastructure, and no new
compression methods were added. Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

## Why KV is Low-Rank

Each layer stores K and V as `[seq, head_dim]` matrices (per kv-head). Across
layers, families, and models, the singular-value energy of these matrices
collapses fast: on TinyLlama the effective (participation) rank is ~2 and ~90%
of K energy sits in ~7 of 64 components (Milestones 5–6). That means most of
the KV feature space is unused — a small orthonormal subspace can carry most of
the signal, and projecting into it is (approximately) lossless.

## How the Subspace Is Built

`scripts/build_kv_subspace.py`:

1. Runs a KV trace: prefill + greedy decode of `TRACE_TOKENS = 64` tokens on
   the real model, capturing the final `DynamicCache`.
2. For **each layer**, reshapes K (and V) to `[kv_heads * seq, head_dim]`
   (heads stacked along the sequence dimension — a *per-layer, head-shared*
   basis) and computes the truncated SVD.
3. The top-`rank` right singular vectors become an **orthonormal** basis
   `W ∈ [head_dim, rank]` for K and separately for V.
4. Saves `reports/kv_subspace/<model>/basis.pt` (tensors + config) and
   `basis_meta.json` (rank, explained variance, one-shot projection error).

Built basis at `rank = head_dim / 4 = 16` (run `20260806_025555`):

| Metric | Value |
|---|---|
| Explained variance (K, rank 16) | 0.8654 |
| Explained variance (V, rank 16) | 0.7003 |
| One-shot K projection error | 0.3651 |
| Layers / kv-heads / head_dim | 22 / 4 / 64 |

**V is visibly less low-rank than K** — a genuinely new observation, since
Milestone 6 only measured K. A static rank-16 basis captures 86.5% of K but
70% of V energy, so V dominates the reconstruction loss.

## How Reuse Works

`scripts/run_kv_subspace_decode.py` loads `basis.pt` once and, after every
decode step, runs:

```python
latent_k = k @ W_k[i]            # project into the fixed subspace
k_recon  = latent_k @ W_k[i].t() # reconstruct before attention
```

and the same for V. The reconstructed K/V replace the cache before the next
forward pass. Because `W` is orthonormal (`W^T W = I`), `W W^T` is an
orthogonal projection, so re-projecting already-projected entries is a no-op —
each new token carries projection loss exactly once. This is the
"reconstruct before attention" variant (operating in the subspace would require
model surgery; reconstruction keeps the decode loop untouched).

## Comparison vs Baseline / Full SVD / Head Pruning

Same greedy decode loop (32 new tokens, prompt `"The capital of France is"`),
rank 16 for svd/subspace, keep 1/4 kv-heads for head pruning.

| Method | Latency (ms) | Effective KV bytes | Break point | Recon error |
|---|---|---|---|---|
| baseline | 10.60 | 833,536 | — | 0.0 |
| svd (recomputed each step) | 126.99 | 208,384 | **None** (identical) | 0.2654 |
| head_prune (1/4 heads) | 11.05 | 208,384 | 1 | 0.9922 |
| **subspace (fixed basis)** | **11.52** | **208,384** | **4** | 0.7575 |

`recon_error` is the mean relative K/V error of the method's final cache versus
the baseline final cache — it conflates pure projection loss with the
downstream content divergence caused by it, so it is a full-pipeline measure.

### The trade-off vs full SVD

- **Full SVD** is the quality ceiling: output is bit-identical to baseline and
  reconstruction error is lowest (0.27), but it costs ~12× the latency (127 ms
  vs 10.6 ms) because it factorizes every layer's K and V every step.
- **Subspace** keeps the low-rank benefit at **baseline latency** (11.5 ms)
  because the expensive factorization is done *once, offline*. It trades the
  static basis's lower variance capture for ~11× speed. The fixed basis
  preserves output through token 4 (vs head pruning's token 1) while cutting KV
  to 1/4.

### Why subspace still outperforms head pruning

Head pruning discards entire kv-heads (exact attention on 1 of 4 heads),
losing a quarter of the cache outright (recon error 0.99, diverges at token 1).
Subspace keeps *all* heads but projects each into a shared low-rank subspace —
a strictly gentler information loss (diverges at token 4).

## Limitations

- **Static basis.** The subspace is learned once from a 64-token trace and
  never updated. KV statistics during a different prompt, longer context, or
  after LoRA/fine-tuning are not represented. A drifting distribution would
  silently raise reconstruction error.
- **No adaptation.** There is no mechanism to refresh the basis or grow its
  rank if the workload changes. Rebuilding is an explicit manual step
  (`scripts/build_kv_subspace.py`).
- **No batching / serving infra.** Decoding is single-sequence, single-model;
  the basis is stored as plain `.pt` tensors, not wired into a serving engine.
- **V is less compressible than K.** A rank-16 static basis captures only 70%
  of V energy; V is the dominant source of error and would need either a higher
  rank or a dedicated basis.
- **Reconstruction is approximate.** Unlike per-step SVD, the fixed basis
  cannot adapt to each step's actual matrix, so output eventually diverges
  (here at token 4 for 32-token decoding).

## Running

```bash
bash scripts/run_kv_subspace_decode.sh   # builds basis, then runs decode
# or, in two steps:
uv run python scripts/build_kv_subspace.py
uv run python scripts/run_kv_subspace_decode.py
```

Success markers:

```text
KV_SUBSPACE_BUILT_OK=1
KV_SUBSPACE_DECODE_OK=1
```

Artifacts:

```text
reports/kv_subspace/<model>/basis.pt
reports/kv_subspace/<model>/basis_meta.json
reports/kv_subspace_decode/run_baseline_<ts>.json
reports/kv_subspace_decode/run_svd_<ts>.json
reports/kv_subspace_decode/run_head_prune_<ts>.json
reports/kv_subspace_decode/run_subspace_<ts>.json
reports/kv_subspace_decode/run_summary_<ts>.json
```

## Relation to Earlier Milestones

- M4 (`docs/04_kv_latent_compare.md`): random projections break output.
- M5 (`docs/05_kv_structured_compare.md`): SVD is lossless but slow.
- M6 (`docs/07_cross_model_validation.md`): low-rank structure generalizes.
- M7 (this doc): the low-rank structure becomes a cached, reusable basis that
  decodes at ~baseline latency.
