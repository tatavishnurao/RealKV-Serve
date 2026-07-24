# Runtime architecture

This document is a template until the target DGX Spark run pins a release and source commit. The final version must link every node to a file and line range in `reports/milestone1/source_map.json` and distinguish direct calls, ownership, inference, and not-observed behavior.

Expected investigation boundary:

```text
LLM.generate
  -> request submission / queue
  -> PyExecutor iteration
  -> scheduler
  -> resource preparation
  -> KVCacheManager
  -> model engine
  -> attention metadata preparation
  -> attention backend forward
  -> paged KV-cache append/read
  -> decoder/sampler
  -> request completion
  -> block release or reuse eligibility
```

No arrow is an observed runtime call until backed by the pinned source map and trace evidence.
