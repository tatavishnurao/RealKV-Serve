# Instrumentation

`realkv/kv_observer.py` is an opt-in append-only JSONL observer. It wraps only methods that have been verified against the exact TensorRT-LLM source commit used by the runtime. It preserves return values and exceptions, records host monotonic timestamps, does not synchronize CUDA, and never copies or logs cache contents.

The current scaffold provides the wrapper primitive. The target-machine execution must populate the source map first and bind the exact `KVCacheManager`, attention metadata, and attention forward methods found in that release. Unsupported observations remain JSON `null`; no inferred block IDs or free-block counts may be substituted.
