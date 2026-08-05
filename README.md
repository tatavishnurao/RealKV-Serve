# RealKV-Serve

Milestone-one compatibility and observability experiment for observing the paged KV-cache lifecycle of one deterministic `Qwen/Qwen3-1.7B` request through the TensorRT-LLM PyTorch backend on one Linux x86_64 workstation with an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM).

This repository owns the experiment harness, schema, reports, and source map. TensorRT-LLM remains an external pinned container dependency. The milestone is observational and does not claim a better cache policy, performance improvement, or production readiness.

## Status

The RTX 4060 is an experimental consumer Ada compatibility target. No NVIDIA support certification, production-serving result, performance comparison, or general TensorRT-LLM compatibility claim is made. Runtime artifacts and acceptance markers are produced only by the local workstation workflow.

## Run on the target workstation

```bash
bash scripts/inspect_host.sh
export REALKV_RUNTIME_IMAGE='nvcr.io/nvidia/tensorrt-llm/release:<verified>@sha256:<resolved>'
bash scripts/pull_runtime.sh
export REALKV_ALLOW_MODEL_DOWNLOAD=1
bash scripts/run_baseline.sh
bash scripts/run_traced_request.sh
bash scripts/validate_milestone1.sh --artifacts
```

See `docs/REPRODUCIBILITY.md` for prerequisites, compatibility boundaries, and the explicit runtime-resolution gate.

## Milestones

| # | Topic | Doc | Marker |
|---|---|---|---|
| 1 | TensorRT-LLM paged KV lifecycle | `docs/ARCHITECTURE.md` | — |
| 2 | Real KV tracing (linear growth) | `docs/02_real_kv_trace.md` | `KV_TRACE_RUN_OK=1` |
| 3 | KV stress experiments (truncation / zeroing / noise) | `docs/03_kv_stress.md` | `KV_STRESS_RUN_OK=1` |
| 4 | Simulated latent KV (random projections) | `docs/04_kv_latent_compare.md` | `KV_LATENT_COMPARE_OK=1` |
| 5 | Structure-preserving KV compression (SVD / head prune / pool) | `docs/05_kv_structured_compare.md` | `KV_STRUCTURED_COMPARE_OK=1` |
| 6 | Cross-model KV-cache validation | `docs/07_cross_model_validation.md` | `CROSS_MODEL_COMPARE_OK=1` |
