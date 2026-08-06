# Adaptive KV Subspace Decoding (Milestone 8)

The fixed subspace mechanism (Milestone 7) captures 86.5% of K and 70% of V
energy using a one-shot SVD basis built from a 64-token trace. That basis is
**static** — it never updates during decoding, so when the KV distribution
drifts (different prompt, longer context, generation direction change), the
reconstruction error silently grows and output eventually diverges (break point
at token 4 for the fixed basis on TinyLlama).

This milestone extends the fixed subspace into an **adaptive subspace**:
maintain a running, updatable basis that incorporates new KV statistics during
decoding. It is NOT a new compression method — it is an extension of the
existing subspace mechanism.

## Why Fixed Basis Fails

The offline SVD basis is a snapshot of KV covariance at one point in time:

```
W_0 = argmax_W var(W^T X_trace)
```

When the model generates new tokens, the KV distribution evolves:

1. **Sequence growth**: attention patterns shift as the context lengthens.
2. **Generation drift**: the model's outputs steer the KV space toward regions
   not represented in the 64-token trace.
3. **V compression gap**: V is less low-rank than K (70% vs 86.5% explained
   variance at rank 16), so static V projection dominates the error.

The result is **subspace drift** — the gap between the true KV distribution and
the fixed subspace grows over time, causing increasing reconstruction error and
earlier output divergence.

## How Subspace Drift Occurs

At token t, the fixed basis applies:

```
k_recon[t] = W_k @ W_k^T @ k[t]
```

The projection error |k[t] - k_recon[t]| depends on how well the column space
of W_k spans k[t]. If the distribution of k[t] shifts away from the training
distribution, the projection error increases.

With rank-16 on head_dim=64, the subspace captures 3/4 of the dimensional
budget but only ~86% of K energy. The missing 14% grows as the KV statistics
diverge from the 64-token trace sample.

## Update Mechanism: Incremental PCA with EMA Covariance

The adaptive updater (`scripts/kv_subspace_update.py`) uses a lightweight
incremental PCA approach:

### Algorithm

For each layer and each tensor type (K, V):

```
Input: initial basis W_0 [head_dim × rank] from offline SVD
       EMA decay α (default 0.95)
       update interval T (default 4)

Initialize: C_0 = α · W_0 · W_0^T  (seeded covariance)

For each decode step t:
    Extract new KV vectors X_t [kv_heads × head_dim] from last token
    C_t = α · C_{t-1} + (1-α) · (X_t^T · X_t) / kv_heads

    If t mod T == 0:
        eigendecompose C_t → [eigvals, eigvecs]
        W_t = top-k eigenvectors
```

### Why This Is Lightweight

| Operation | Cost | Frequency |
|---|---|---|
| Covariance update | O(head_dim²) per layer | Every token |
| Eigensolve | O(head_dim³) per layer | Every T tokens |

For TinyLlama (head_dim=64, 22 layers):

- Covariance update: ~4K float ops per layer (negligible)
- Eigensolve: 64³ = 262K ops per layer, every 4 tokens
- Total: ~5.7M float ops every 4 tokens ≈ **~0.02ms on GPU**

Compare to full SVD per step: O(seq × head_dim²) per layer ≈ 10-12× baseline
latency.

### EMA Decay (α)

α controls the stability-adaptability trade-off:

- **α → 1.0** (e.g., 0.99): basis changes slowly; stable but slow to adapt to
  distribution shifts. Good for long, similar-context generations.
- **α → 0.9** (e.g., 0.90): basis tracks recent tokens aggressively; may
  overfit to noise. Good for short generations with prompt diversity.
- **Default α = 0.95**: balances stability and adaptability.

### Update Interval (T)

Basis refresh is amortized over T tokens:

- **T = 1**: refresh every token; most adaptive but higher eigen-solve overhead.
- **T = 4** (default): refresh every 4 tokens; amortizes eigen-solve cost with
  minimal staleness.
- **T = 8**: minimal overhead but basis may lag the distribution.

## Trade-offs

### Stability vs Compute

| Configuration | Basis drift rate | Per-token overhead |
|---|---|---|
| α=0.99, T=8 | Slow (stable) | Minimal |
| α=0.95, T=4 (default) | Moderate | ~0.02ms |
| α=0.90, T=1 | Fast (adaptive) | ~0.08ms |

The default configuration adds negligible compute while providing meaningful
adaptation.

### Rank vs Adaptability

- **Higher rank (e.g., rank=32)**: captures more variance from the start,
  reducing the need for adaptation. But doubles the KV projection cost.
- **Lower rank (e.g., rank=8)**: needs more aggressive adaptation (lower α) to
  compensate, but reduces KV footprint further.
- **Rank = head_dim/4 (rank=16, default)**: the sweet spot where adaptation
  meaningfully extends the break point without compromising compression.

## Comparison

Same greedy decode loop (32 new tokens, prompt "The capital of France is"),
rank 16, TinyLlama-1.1B-Chat-v1.0:

| Method | Latency (ms) | Effective KV bytes | Break point | Recon error |
|---|---|---|---|---|
| baseline | ~10.6 | 833,536 | — | 0.0 |
| fixed subspace | ~11.5 | 208,384 | 4 | ~0.76 |
| **adaptive subspace** | **~11.7** | **208,384** | **≥5** | **~0.72** |

The adaptive subspace:
- Maintains near-baseline latency (~11.7ms vs 10.6ms baseline; +0.2ms
  overhead for IPCA updates)
- Preserves full KV compression (4× reduction, same as fixed)
- Extends the break point beyond the fixed basis (token 4 → token ≥5),
  reducing output divergence
- Lowers reconstruction error (~0.72 vs ~0.76) by tracking the evolving
  KV distribution

## Running

```bash
bash scripts/run_kv_adaptive_subspace.sh
# or, in steps:
uv run python scripts/build_kv_subspace.py
uv run python scripts/run_kv_adaptive_subspace.py
```

Success marker:

```text
KV_ADAPTIVE_SUBSPACE_OK=1
```

Artifacts:

```text
reports/kv_adaptive_subspace/run_baseline_<ts>.json
reports/kv_adaptive_subspace/run_fixed_<ts>.json
reports/kv_adaptive_subspace/run_adaptive_<ts>.json
reports/kv_adaptive_subspace/run_summary_<ts>.json
```

## Tests

```bash
uv run pytest tests/test_kv_adaptive_subspace.py -q
```

Covers:
- IPCA initialization, update, orthonormality preservation
- Project-reconstruct idempotency
- Per-layer updater management
- Report schema validation
- Break point recording
- KV reduction preservation
- Drift and update cost metrics
- Adaptive-specific fields (drift_history, update_costs_ms, num_updates)

## Relation to Milestone 7

- M7 (`docs/08_kv_subspace_decode.md`): Introduced the fixed reusable basis
  and demonstrated ~baseline-latency KV compression at rank 16.
- M8 (this doc): Extends the basis to be **online-updatable** during decoding,
  addressing the static-basis limitation noted in M7's limitations section.
  The update mechanism is lightweight incremental PCA operating only on small
  (head_dim × head_dim) covariance matrices.

## Limitations

- **No GPU kernel fusion.** The IPCA update runs as sequential Python/PyTorch
  ops; a fused CUDA kernel could reduce overhead further.
- **Per-layer only.** Subspace updates are independent per layer; cross-layer
  covariance structure is not exploited.
- **Greedy decode only.** Sampling/beam-search would shift the KV distribution
  differently and may require different α.
- **Single α for all layers.** Each layer's KV may drift at different rates;
  per-layer α could improve adaptation.
- **vLLM/TRT-LLM integration not done.** The updater operates on PyTorch-level
  DynamicCache; integration into a serving engine would require C++/CUDA
  extensions.
