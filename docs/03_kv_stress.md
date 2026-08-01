# KV-Cache Stress Experiments (Milestone 3)

Controlled perturbations of the real KV cache during greedy decoding on
**TinyLlama/TinyLlama-1.1B-Chat-v1.0** (PyTorch + HuggingFace, `use_cache=True`).

Goal: measure **output divergence**, **KV size behavior**, and **latency impact**
when the cache is truncated, zeroed, or noised — not infrastructure work.

## What Each Manipulation Simulates

| Experiment | Parameter | What it simulates |
|---|---|---|
| **Truncation (full)** | `full` | Baseline: complete cache, normal decode |
| **Truncation (half)** | `half` | Aggressive sliding window: keep only the most recent half of positions each step |
| **Truncation (last 8)** | `8` | Extremely short context window (last 8 tokens only) |
| **Zeroing** | step `10` | Catastrophic cache wipe / hard memory corruption mid-generation |
| **Noise** | `ε=1e-4` | Mild numerical noise / quantization-like error, applied every step |
| **Noise** | `ε=1e-2` | Strong noise; tests whether error accumulates over decode steps |

### Truncation

```python
past_key_values = tuple(
    (k[:, :, -N:, :], v[:, :, -N:, :])
    for (k, v) in past_key_values
)
```

Keeps only the last `N` sequence positions. Models with RoPE then attend over a
shorter history with position mismatch — quality should drop as `N` shrinks.

### Zeroing

At a fixed step (default **10**), every key and value tensor is set to zero.
Later tokens still append new non-zero entries, but all prior context is gone.
This is the harshest discrete failure mode.

### Noise injection

```python
k += torch.randn_like(k) * epsilon
v += torch.randn_like(v) * epsilon
```

Applied **after every decode step** so perturbations can compound. Small `ε`
may be nearly invisible; large `ε` should derail generation.

## Metrics Captured

For each run, JSON reports under `reports/kv_stress/` include:

- `experiment` — type (`truncation` / `zeroing` / `noise`)
- `param` — mode or magnitude
- `output` — final generated text
- `tokens_generated` — count of new tokens (~32)
- `latency_ms` — average per-step latency
- `kv_bytes` — total KV tensor bytes at the final step
- `divergence` — comparison string vs full-cache baseline

## Expected vs Observed Behavior

### Which breaks output the most?

**Expectation (ordered severity):**

1. **Zeroing at step 10** — should break mid-sequence continuity hardest
2. **Noise ε=1e-2** — strong accumulated corruption
3. **Truncation last-8** — severe context loss
4. **Truncation half** — moderate degradation
5. **Noise ε=1e-4** — often near-baseline
6. **Full baseline** — reference

**Observation:** Re-run `bash scripts/run_kv_stress.sh` and compare `output` /
`divergence` fields in `reports/kv_stress/*.json`. In practice, zeroing and
large-ε noise produce the largest text divergence; last-8 truncation also
diverges sharply from the full-cache continuation.

### Does truncation degrade gracefully?

**Expectation:** Yes — full ≈ half > last-8 in quality and fidelity to baseline.

**Observation:** Half truncation often preserves a longer common prefix with the
baseline than last-8. Last-8 tends to diverge earlier because prompt+history
beyond 8 tokens is discarded every step. KV byte counts at the end of a last-8
run stay near a small plateau (cache cannot grow unbounded past ~8 positions
between steps before the next truncate).

### Does noise accumulate?

**Expectation:** Yes when noise is injected every step — variance in the cache
grows with depth.

**Observation:** `ε=1e-4` often remains close to baseline for short 32-token
runs. `ε=1e-2` typically yields clearly different text, consistent with
step-wise accumulation rather than a one-shot perturbation.

### Latency impact

Manipulations are tensor ops on the cache between forward passes. On short
TinyLlama runs they add little wall-time versus the forward itself; average
`latency_ms` stays in the same order of magnitude as baseline. Latency is
recorded so regressions remain measurable on RTX 4060 (or CPU fallback).

## Running

```bash
bash scripts/run_kv_stress.sh
# or
python scripts/run_kv_stress.py
```

Success marker:

```text
KV_STRESS_RUN_OK=1
```

Reports:

```text
reports/kv_stress/truncation_full_<timestamp>.json
reports/kv_stress/truncation_half_<timestamp>.json
reports/kv_stress/truncation_last8_<timestamp>.json
reports/kv_stress/zeroing_step10_<timestamp>.json
reports/kv_stress/noise_eps1e-4_<timestamp>.json
reports/kv_stress/noise_eps0p01_<timestamp>.json
reports/kv_stress/summary_<timestamp>.json
```

## Relation to Milestone 2

Milestone 2 (`docs/02_real_kv_trace.md`) measured **healthy** linear KV growth.
Milestone 3 **stresses** that cache deliberately to map failure modes relevant
to paged/ truncated/ lossy KV serving designs.
