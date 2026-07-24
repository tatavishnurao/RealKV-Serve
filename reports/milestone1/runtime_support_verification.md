# Runtime support verification

Verification date: 2026-07-20. This is a source record, not a claim that the target runtime was executed.

The official TensorRT-LLM Qwen model guide documents Qwen3-8B and states that Qwen3 uses the PyTorch backend: <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md>. The official Qwen3 project gives the PyTorch serving form `trtllm-serve Qwen/Qwen3-8B --backend pytorch`: <https://github.com/QwenLM/Qwen3>. The official TensorRT-LLM release page currently shows DGX-Spark test coverage in release material: <https://github.com/NVIDIA/TensorRT-LLM/releases>.

The exact NGC image tag, digest, ARM64/SBSA/GB10 support assertion, package versions, and source commit are unresolved until NGC can be queried and the container is inspected on the DGX Spark. Therefore this repository intentionally does not set a default image, does not claim Qwen3-8B FP16 is validated on the target, and does not run inference from this host. `scripts/pull_runtime.sh` refuses an unpinned image.

Status:

```text
SINGLE_DGX_SPARK_CONFIRMED=UNAVAILABLE
TENSORRT_LLM_PYTORCH_BACKEND_CONFIRMED=SUPPORTED_BY_OFFICIAL_MODEL_GUIDE_NOT_RUNTIME_VERIFIED
QWEN3_8B_FP16_TARGET_VALIDATION=UNAVAILABLE
ARM64_GB10_CONTAINER_VALIDATION=UNAVAILABLE
```
