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

Measured on **RTX 4060**, TinyLlama-1.1B, greedy, 32 new tokens, prompt
`"The capital of France is"` (run `20260801_074756`).

### Baseline output (full KV)

```text
Paris.

2. B. The capital of Germany is Berlin.

3. C. The capital of the United States is Washington, D.
```

### Which breaks output the most?

**Expectation (ordered severity):**

1. **Zeroing at step 10** — break mid-sequence continuity hardest
2. **Noise ε=1e-2** — strong accumulated corruption
3. **Truncation last-8 / half** — severe context loss
4. **Noise ε=1e-4** — near-baseline
5. **Full baseline** — reference

**Observation (this run):**

| Experiment | Divergence vs baseline | Notes |
|---|---|---|
| half truncation | differs early (`common_prefix_chars=5`) | Degenerates to `"Paris is isis,"` then newlines |
| last-8 truncation | differs (`common_prefix_chars=8`) | Keeps `"Paris.\n\n"` then garbage / colon spam |
| zeroing @ step 10 | differs (`common_prefix_chars=29`) | Preserves early quiz structure, then `"and and and..."` loop |
| noise ε=1e-4 | **identical** | Below damage threshold for 32 tokens |
| noise ε=1e-2 | differs (`common_prefix_chars=8`) | Diverts into a different instructional continuation |

**Most disruptive in practice:** aggressive **half truncation** produced the
most unusable text. **Zeroing** was the harshest *discrete* mid-run failure
(long shared prefix, then total collapse). **ε=1e-2 noise** broke semantics
without full gibberish. **ε=1e-4** did not break output at all.

### Does truncation degrade gracefully?

**Expectation:** Yes — full ≳ half > last-8 in fidelity.

**Observation:** Not fully graceful. **Half** truncation was *worse* than
**last-8** here: half collapsed into repetitive whitespace, while last-8 still
emitted token-like structure. Both diverged from baseline quickly. Final
`kv_bytes` for half (~22 KiB) and last-8 (~176 KiB) stayed far below full
(~814 KiB), confirming the cache cannot grow unboundedly under per-step
windowing.

### Does noise accumulate?

**Expectation:** Yes when injected every step.

**Observation:** Supported. Small noise (`1e-4`) matched baseline exactly over
32 steps. Large noise (`1e-2`) changed the continuation after a short shared
prefix — consistent with per-step accumulation crossing a quality threshold
rather than a single-shot flip.

### Latency impact

| Experiment | Avg latency (ms) |
|---|---|
| full | 114.2 |
| half | 61.1 |
| last-8 | 65.0 |
| zeroing | 47.8 |
| noise 1e-4 | 75.0 |
| noise 1e-2 | 74.8 |

Manipulations are cheap relative to the forward pass. Shorter effective cache
(truncation) correlated with *lower* average step latency in this run. Numbers
are host-observed averages, not kernel timers.

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
