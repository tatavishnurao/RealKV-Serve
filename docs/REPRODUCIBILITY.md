# Reproducibility

Run from the repository root or use the scripts from any working directory; each resolves its own root. The target is one ARM64/SBSA DGX Spark with GB10 unified memory and one visible device. Do not use `sudo`, alter drivers/CUDA/firmware, or reuse an unpinned TensorRT-LLM image.

Before downloading weights, set `REALKV_ALLOW_MODEL_DOWNLOAD=1`. Without it, the workload must fail closed and report `Qwen/Qwen3-8B` and the configured cache path. Set `REALKV_MODEL_CACHE` to a project-local cache. Plan for approximately 16 GB of FP16 weights plus tokenizer/config and runtime/model working memory; actual requirements are release-dependent and must be recorded from the run.

The image must be an NGC TensorRT-LLM release tag resolved to a digest and officially verified for DGX Spark/GB10 ARM64, Qwen3-8B FP16, and the PyTorch backend. `pull_runtime.sh` refuses a tag without `@sha256:`. The resolved image inspect output, package versions, model revision, and upstream commit belong in `runtime_environment.json`.

The scripts intentionally do not claim container GPU access from host `nvidia-smi`; `inspect_host.sh` runs a separate container probe. A fresh run must capture exact command, timestamps, exit status, stdout, and stderr in `COMMAND_OUTPUT_LOG.md`, with credentials and tokens redacted.
