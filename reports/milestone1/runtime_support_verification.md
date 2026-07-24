# Runtime support verification

Verification date: 2026-07-20. This is a source record, not a claim that the target runtime was executed.

The official TensorRT-LLM Qwen model guide documents Qwen3 usage and the PyTorch backend: <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md>. This source record does not establish named validation for the GeForce RTX 4060 Laptop GPU or for the selected Qwen3-1.7B workload.

The exact NGC image tag, digest, package versions, and source commit remain unresolved. Host CUDA access is present, but the independent Docker GPU probe failed before container startup with `failed to discover GPU vendor from CDI: no known GPU vendor found`. Therefore this repository does not claim container compatibility, local TensorRT-LLM compatibility, or model execution. `scripts/pull_runtime.sh` refuses an unpinned image.

Status:

```text
RTX_4060_CONFIRMED=1
CONTAINER_GPU_ACCESS_CONFIRMED=FAILED
TENSORRT_LLM_PYTORCH_BACKEND_CONFIRMED=UNAVAILABLE
QWEN3_1_7B_TARGET_VALIDATION=UNAVAILABLE
```
