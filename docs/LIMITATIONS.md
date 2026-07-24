# Limitations and non-claims

The RTX 4060 Laptop GPU is an 8 GB consumer Ada device used here only as a compatibility target. Generic `nvidia-smi` framebuffer usage is useful context but not an authoritative substitute for runtime allocation data. Record `torch.cuda.memory_allocated`, `torch.cuda.memory_reserved`, process RSS, and system available memory separately; never combine them into one GPU-memory number.

This experiment does not establish official TensorRT-LLM validation for the GeForce RTX 4060. It must distinguish documented architecture support, named officially validated hardware, container GPU visibility, local runtime compatibility, and successful execution of the exact pinned model.

This milestone does not prove a better KV-cache policy, improved throughput, lower TTFT/ITL, production readiness, multi-request correctness, concurrency behavior, prefix-reuse effectiveness, eviction quality, precision-format model-quality preservation, or superiority over FlashInfer, vLLM, or the unmodified runtime. Later phases such as concurrency, prefix reuse, block-size sweeps, FP8 KV cache, backend comparisons, and custom policies are explicitly out of scope.
