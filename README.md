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
