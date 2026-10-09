# Gate-anchor router-only speed comparison

**Checkpoint:** `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`  
**Completed:** 2026-10-09 02:22:33 America/New_York  
**Device:** NVIDIA GeForce RTX 5090 (`cuda:0`), Windows, PyTorch 2.11.0+cu128

## Result

For one token's 40 layer routing decisions, the native gate took a median **21.80 ms** and the gate-anchor ANN router took **105.94 ms**. The redesigned router therefore added **84.14 ms per token** for routing, or **3.86 times more latency** (4.86 times the native routing latency).

| Router | Median per layer decision | Median across 40 layers for one token |
|---|---:|---:|
| Native 256-way gate | 545 μs | 21.80 ms |
| Gate-anchor ANN, 32 candidates | 2,648 μs | 105.94 ms |
| Additional gate-anchor time | 2,103 μs | 84.14 ms |

There were three alternating timing trials per router, each covering 64 contexts and 2,560 layer decisions. Individual total times were 1.382–1.433 seconds for native and 6.651–6.810 seconds for gate-anchor. Across 640 parity checks, both routers selected the same ordered top-eight IDs.

## What the timing includes

The benchmark loaded saved BF16 native validation hidden states and the quantized gate weights embedded in the rebuilt anchor bank. For each context it timed all 40 routing calls: gate projection, candidate retrieval and reranking, softmax, top-eight selection, and normalization. It synchronized CUDA after each layer decision, so each measurement waits for the GPU routing result and includes the anchor router's CPU HNSW query and GPU/CPU transfer.

It excludes all expert MLP computation, shared-expert computation, attention, embedding, model loading, and token generation. The result is routing-decision latency for this 40-layer, 256-expert model, not end-to-end token latency. Per-layer synchronization makes each isolated decision complete before timing the next; whole-model execution can overlap GPU work differently. This is a router-only comparison, not a general inference throughput claim.

## Interpretation

The gate-anchor index preserved native routing decisions on the sampled hidden states, but the current Python/CPU HNSW path is slower than the native GPU gate calculation. Its measured cost is chiefly the extra projection and the GPU-to-CPU query transfer, CPU search, and return of candidates for GPU reranking. At this expert count, the native gate is inexpensive enough that the fidelity control has no speed advantage.

The measurements suggest a concrete optimization target if ANN routing remains necessary: keep query projection and retrieval on the GPU, avoid a host round-trip for every layer/token, and benchmark an implementation that batches or pipelines candidate searches. These optimizations have not been implemented or measured.
