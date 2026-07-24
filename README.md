# RealKV-Serve

Milestone-one experiment for observing the paged KV-cache lifecycle of one deterministic `Qwen/Qwen3-8B` request through the TensorRT-LLM PyTorch backend on one DGX Spark.

This repository owns the experiment harness, schema, reports, and source map. TensorRT-LLM remains an external pinned container dependency. The milestone is observational and does not claim a better cache policy, performance improvement, or production readiness.

## Status

The host used to scaffold this repository is not a DGX Spark and no model/container execution has been performed here. Runtime artifacts and acceptance markers must be produced only by the target-machine workflow.

## Run on DGX Spark

```bash
bash scripts/inspect_host.sh
export REALKV_RUNTIME_IMAGE='nvcr.io/nvidia/tensorrt-llm/release:<verified>@sha256:<resolved>'
bash scripts/pull_runtime.sh
export REALKV_ALLOW_MODEL_DOWNLOAD=1
bash scripts/run_baseline.sh
bash scripts/run_traced_request.sh
bash scripts/validate_milestone1.sh --artifacts
```

See `docs/REPRODUCIBILITY.md` for prerequisites and the explicit runtime-resolution gate.
