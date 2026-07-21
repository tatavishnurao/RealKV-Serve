# Limitations and non-claims

DGX Spark GB10 uses unified memory. Generic `nvidia-smi` framebuffer usage may be unavailable or misleading, so it is not the authoritative memory metric. Record `torch.cuda.memory_allocated`, `torch.cuda.memory_reserved`, process RSS, and system available memory separately; never combine them into one GPU-memory number.

This milestone does not prove a better KV-cache policy, improved throughput, lower TTFT/ITL, production readiness, multi-request correctness, concurrency behavior, prefix-reuse effectiveness, eviction quality, precision-format model-quality preservation, or superiority over FlashInfer, vLLM, or the unmodified runtime. Later phases such as concurrency, prefix reuse, block-size sweeps, FP8 KV cache, backend comparisons, and custom policies are explicitly out of scope.
