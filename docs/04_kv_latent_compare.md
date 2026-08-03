# Simulated Latent KV vs Baseline Full KV (Milestone 4)

Compares **full KV caching** against a **simulated latent KV** representation
inside the same real decoding loop on
**TinyLlama/TinyLlama-1.1B-Chat-v1.0** (PyTorch + HuggingFace, `use_cache=True`).

Goal: quantify output divergence, latency, and KV memory when each layer's K/V
is passed through a linear latent bottleneck instead of being stored in full —
a cheap software simulation of a latent KV / MLA-style design (no CUDA kernels).

## What Is Being Simulated

Baseline stores full K/V tensors in the cache:

```python
outputs = model(input_ids=..., use_cache=True)
past_key_values = outputs.past_key_values
```

Latent mode replaces the cache every step with a lossy reconstruction:

```python
latent_k = k @ W_down            # [head_dim] -> [latent_dim]
latent_v = v @ W_down
k_recon  = latent_k @ W_up       # [latent_dim] -> [head_dim]
v_recon  = latent_v @ W_up
past_key_values[i] = (k_recon, v_recon)
```

`W_down` is a **seeded orthonormal** `[head_dim, latent_dim]` matrix and
`W_up = W_down^T`, so `W_down @ W_up` is an orthogonal projection onto a random
`latent_dim`-dimensional subspace. Because an orthogonal projection is
idempotent, already-compressed positions are *not* re-damaged on later steps —
only each newly appended KV entry carries compression loss, exactly like a real
latent KV system. `latent_dim = head_dim // compression_ratio`
(`head_dim = 64` here).

## Metrics Captured

Each JSON report under `reports/kv_latent_compare/` includes:

- `mode` — `baseline` | `latent`
- `compression_ratio` — `1` (baseline) or `2 / 4 / 8`
- `output` — final generated text
- `tokens_generated` — new tokens (~32)
- `latency_ms` — average per-step latency (host-observed, includes projection)
- `kv_bytes` — effective compressed KV footprint (latent bytes for `latent`,
  full KV bytes for `baseline`)
- `kv_bytes_stored` — bytes physically held in the cache (reconstructed tensors)
- `divergence` — comparison string vs baseline output
- `common_prefix_chars` — shared prefix length vs baseline
- `kv_saved_bytes` — `baseline_kv_bytes - kv_bytes`

## Results (run `20260803_153838`, RTX 4060)

| Mode | Ratio | Latency (ms) | KV bytes (footprint) | Common prefix chars | Output |
|---|---|---|---|---|---|
| baseline | 1 | 11.35 | 833,536 | — | `Paris.\n\n2. B. The capital of Germany is Berlin.\n\n3. C. The capital of the United States is Washington, D.` |
| latent | 2 | 11.43 | 416,768 | 5 | `Paris, the capital of the capital of the of the of ...` |
| latent | 4 | 12.03 | 208,384 | 5 | `Paris, and the the the the the the way of the the ...` |
| latent | 8 | 11.54 | 104,192 | 7 | `Paris.\n Резуpays.\nWHERE, and and and and and and ...` |

### How much divergence occurs at each compression ratio?

Substantial at **every** ratio. All latent runs break away from the baseline
within the first handful of characters (`common_prefix_chars ∈ {5, 7}`) and
collapse into repetitive token loops. The shared prefix ("`Paris,`") survives,
but the instructive "quiz" continuation that the baseline produces is gone in
all three latent runs.

Measured relative K reconstruction error on the real cache (averaged over all
layers) explains why:

| Ratio | latent_dim | mean relative ‖K_recon − K‖ / ‖K‖ |
|---|---|---|
| 2 | 32 | 0.70 |
| 4 | 16 | 0.87 |
| 8 | 8 | 0.94 |

A *random* orthonormal projection discards 50–88% of each KV vector's energy, so
even 2× compression is far too lossy for coherent generation.

### Is degradation gradual or catastrophic?

**Catastrophic, and nearly flat across ratios.** There is no graceful curve:
2× already destroys output coherence, and 4×/8× are not meaningfully worse at
the text level (all share a 5–7 char prefix then degrade). The recovery of a
longer prefix at 8× (`7` chars) than at 2×/4× (`5`) is a quirk of the specific
seeded projection, not evidence of non-monotonic quality.

A lossless sanity check confirms the mechanism is correct: at `ratio = 1`
(full-rank projection) the latent run reproduces the baseline text **exactly**.
Divergence therefore comes purely from the projection bottleneck.

### Does latency increase?

**No.** Latency is flat within noise (11.35 ms baseline vs 11.43–12.03 ms
latent). The two small matmuls per layer are negligible against the forward
pass. Note that in a real latent KV system the payoff would be in *stored*
memory (and paging), not in step latency.

### Does KV memory reduce meaningfully?

**Yes, and proportionally.** The effective footprint scales exactly with the
ratio: 833,536 → 416,768 (2×) → 208,384 (4×) → 104,192 (8×) bytes, i.e. a
guaranteed `1 / ratio` reduction. This is the one outcome that matches
expectation precisely, because the latent dimension is defined to be
`head_dim / ratio`. (`kv_bytes_stored` stays at 833,536 because this simulation
still materializes full-size reconstructed tensors in the PyTorch cache.)

### Does this validate your latent KV idea?

**Partially, with an important caveat.** It validates that the *mechanics* work
end-to-end: KV can be projected to a latent space, reconstructed, fed back into
the cache, and memory scales down by exactly the ratio. But it also shows that
**naive random linear projections are not usable** — a real latent KV system
must *learn* `W_down`/`W_up` from the actual KV distribution (e.g. via a
low-rank / SVD / autoencoder objective on traced K/V, as in MLA) so that
reconstruction error is a few percent, not 70%+. That learned projection is a
natural follow-up milestone; this one deliberately uses the simplest possible
linear map to isolate the cost of compression itself.

## Running

```bash
bash scripts/run_kv_latent_compare.sh
# or
python scripts/run_kv_latent_compare.py
```

Success marker:

```text
KV_LATENT_COMPARE_OK=1
```

Reports:

```text
reports/kv_latent_compare/baseline_<timestamp>.json
reports/kv_latent_compare/latent_ratio2_<timestamp>.json
reports/kv_latent_compare/latent_ratio4_<timestamp>.json
reports/kv_latent_compare/latent_ratio8_<timestamp>.json
reports/kv_latent_compare/summary_<timestamp>.json
```

## Relation to Earlier Milestones

- Milestone 2 (`docs/02_real_kv_trace.md`): healthy linear KV growth (baseline).
- Milestone 3 (`docs/03_kv_stress.md`): destructive cache perturbations.
- Milestone 4: a *lossy representation* of the cache — the first step toward a
  real latent / compressed KV serving design.
