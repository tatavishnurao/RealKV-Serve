# Real KV-Cache Lifecycle Trace

## What is the KV Cache?

In autoregressive transformer decoding, the key-value (KV) cache stores the
key and value tensors from all previous attention layers. Instead of
recomputing the full attention over the entire sequence at each step, the
model can reuse the cached representations from prior tokens and only compute
attention for the newly generated token.

## Observed Growth Pattern

For TinyLlama-1.1B (22 layers, 32 heads, hidden dim 2048, dtype float16):

- Each KV layer stores: `[2, batch, num_heads, seq_len, head_dim]`
- For each new token, KV cache grows by:
  `22 layers × 2 (K+V) × batch × num_heads × 1 × head_dim × 2 bytes`
- Growth per token: ~22 × 2 × 1 × 32 × 1 × 64 × 2 = **180,224 bytes**
- Total at 32 tokens: ~5.6 MiB (negligible vs 8 GB VRAM)

## Memory vs Tokens

KV cache size increases linearly with sequence length:

```text
SEQ_LEN  KV_BYTES
    3     541,696
    4     722,944
    5     904,192
    ...   linear growth ...
   34    ~5,766,144
```

## Latency Per Token Trend

The first token (prefill) has higher latency due to full-sequence attention.
Subsequent decode tokens show stable, flat latency since only the new token
requires attention computation. TinyLlama-1.1B on RTX 4060 typically achieves
~30-50 ms per decode token.

## Differences from Synthetic Repo (LatentPagedAttention-rs)

| Aspect | LatentPagedAttention-rs (synthetic) | RealKV-Serve Milestone 2 (real) |
|---|---|---|
| Model | No real model; synthetic kernel | TinyLlama-1.1B via HuggingFace |
| KV source | Simulated block-level allocator | Real `past_key_values` from `transformers` |
| Attention | Custom paged kernel stub | PyTorch native scaled dot-product |
| Memory | Hypothetical block budgets | Real `torch.cuda.memory_allocated()` |
| Batching | Simulated multi-request | Single request, single GPU |
| Precision | Assumed FP16 | Actual FP16 (RTX 4060) |
| Latency | Not measured | Real per-token latency |

## Running

```bash
python scripts/run_kv_trace.py
```

Output: `reports/kv_trace/run_<timestamp>.jsonl`
