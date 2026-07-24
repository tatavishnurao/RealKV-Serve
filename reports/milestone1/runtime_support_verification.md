# Runtime support verification

Verification date: 2026-07-20. This is a source record, not a claim that the target runtime was executed.

The TensorRT-LLM Qwen model guide and Qwen project documentation are source references for the current PyTorch execution path. The active experiment will assess Qwen3-1.7B on one RTX 4060 Laptop GPU only after a release, immutable image digest, and exact model revision are pinned.

The exact NGC image tag, digest, architecture-support assertion, named official-validation status, package versions, and source commit are unresolved until NGC is queried and the container is inspected on the target workstation. Therefore this repository intentionally does not set a default image, does not claim Qwen3-1.7B FP16 is validated on the target, and does not run inference from this host. `scripts/pull_runtime.sh` refuses an unpinned image.

Status:

```text
RTX_4060_CONFIRMED=UNAVAILABLE
TENSORRT_LLM_PYTORCH_BACKEND_CONFIRMED=UNAVAILABLE
QWEN3_1_7B_TARGET_VALIDATION=UNAVAILABLE
RTX_4060_CONTAINER_VALIDATION=UNAVAILABLE
```
