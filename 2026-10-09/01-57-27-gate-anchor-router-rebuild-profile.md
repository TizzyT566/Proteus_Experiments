# Gate-anchor router rebuild profile

**Checkpoint:** `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`  
**Build completed:** 2026-10-09 01:57:27 America/New_York  
**Platform:** Windows, Python 3.12, PyTorch 2.11.0+cu128, RTX 5090 selected as `cuda:0`  
**Result:** successful, exit code 0

## Measured wall time

The complete router-builder process took **69.406 seconds** (1 minute 9.4 seconds), from immediately before Python launch through process exit. It read the existing quantized gate weights, generated anchors and 40 per-layer HNSW indexes, sampled and validated 256 held-out hidden states per layer, and wrote the bank and report. The process timer includes Python startup, imports, file reads and writes, index construction, and the builder's sample validation. It excludes model quantization, full-model loading, and the separate 180,160-position exhaustive validator.

This is the measured time for rebuilding all 40 layers with 256 experts per layer on this machine. It is not a measurement of an incremental registry add/remove. The router builder also dequantizes each layer's gate matrix to construct the anchors; it does not re-quantize or rebuild the model.

## Configuration and result

The router used the quantized native gate weights, a per-layer QR gate rowspace, equal-norm lifting into 384D vectors, squared-L2 HNSW, and 32 retrieved candidates followed by native-logit reranking. Each of 40 layers was checked against 256 held-out hidden states (10,240 layer-token observations total).

| Metric | Mean across layers |
|---|---:|
| Native top-eight experts present in the candidate set | 1.000000 |
| Top-eight recall | 1.000000 |
| Exact top-eight set | 1.000000 |
| Routing-weight L1 difference | 1.0281e-6 |

All 40 layers completed with full candidate support on their 256 sampled states. This builder check is a smaller sample than the prior full validation of 4,504 positions per layer; that earlier full validation remains the stronger coverage result.

## Artifacts

The rebuilt bank and indexes are in `results/gate_anchor_router_qwen36_int4_l2_rebuild_20261009/`. They contain 42 files totaling 117,999,625 bytes. The prior validated router directory was left untouched. The wall-time record and complete layer-by-layer console output are in `artifacts/runlogs/gate_anchor_router_rebuild_20261009.profile.json` and `artifacts/runlogs/gate_anchor_router_rebuild_20261009.out.log`.

## Interpretation

For this 40-layer, 256-expert checkpoint, a full offline rebuild took about 69 seconds and reproduced the gate decisions on its build-time sample. A registry service could rebuild only the affected layer in a background worker and switch to the finished index after validation. The current script rebuilds all layers and has fixed Qwen dimensions; it does not yet provide variable expert counts or online add/remove operations. Those need a registry-aware builder and a separate latency benchmark.
