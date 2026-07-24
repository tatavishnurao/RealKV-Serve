# Reproducibility

Run from the repository root or use the scripts from any working directory; each resolves its own root. The active target is one Linux x86_64 workstation with one NVIDIA GeForce RTX 4060 Laptop GPU and approximately 8 GB VRAM. Do not use `sudo`, alter drivers/CUDA/firmware, or reuse an unpinned TensorRT-LLM image.

Before downloading weights, set `REALKV_ALLOW_MODEL_DOWNLOAD=1`. Without it, the workload must fail closed and report `Qwen/Qwen3-1.7B` and the configured cache path. Set `REALKV_MODEL_CACHE` to a project-local cache. The 8 GB device is a bounded compatibility target, so actual model and KV-cache memory requirements must be captured from the run.

The image must be an NGC TensorRT-LLM release tag resolved to a digest. Documented architecture support, named officially validated GPU models, container GPU visibility, actual local runtime compatibility, and successful execution must be recorded as separate facts. `pull_runtime.sh` refuses a tag without `@sha256:`. The resolved image inspect output, package versions, model revision, and upstream commit belong in `runtime_environment.json`.

The scripts intentionally do not claim container GPU access from host `nvidia-smi`; `inspect_host.sh` runs a separate container probe. A fresh run must capture exact command, timestamps, exit status, stdout, and stderr in `COMMAND_OUTPUT_LOG.md`, with credentials and tokens redacted.
