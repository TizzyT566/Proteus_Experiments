# Contribution-weighted 2,048D expert vectors

**Completed:** 2026-10-09 06:55:31 Eastern Daylight Time  
**Checkpoint:** `Qwen/Qwen3.6-35B-A3B` at `995ad96eacd98c81ed38be0c5b274b04031597b0` (the selected post-trained checkpoint)  
**Environment:** Windows, Python 3.12, PyTorch CUDA; RTX 5090 (`cuda:0`) used for vector aggregation and exhaustive scoring. USearch HNSW lookup ran on CPU.

## Result

The implementation creates one full-width vector for each layer/expert by averaging the captured hidden states in proportion to the expert's native routing weight. It performs **no HDBSCAN clustering and no PCA**. The 40-layer bank contains 10,240 vectors, each 2,048D.

The weighted mean is a small improvement over an unweighted mean, but it does **not** make a reliable expert router by itself. Across 180,160 held-out layer-token observations, the weighted HNSW router achieved **0.5960 top-8 recall**, **0.0192 exact-set rate**, and **0.9740 routing-weight L1**. The previous HDBSCAN/PCA-384 ANN router on the same corpus/split achieved 0.6938, 0.2404, and 0.8272 respectively. HNSW closely matched exhaustive L2 scoring, so the poor result is primarily the single-mean scoring model, not neighbor-search approximation.

The INT4 live generation check also failed quality-wise: both 48-token runs returned repetitive, confused text about PCA and nearest-neighbor routing. A 16-token minimum did not improve the answer.

## Vector construction and evaluation

For layer l and expert e, let S(e) be training token positions where e appears in the native top eight. Let alpha(t,e) be the corresponding native top-eight mixture weight. The vector is:

```text
v(e,l) = sum over t in S(e) [alpha(t,e) * h(t,l)] / sum over t in S(e) [alpha(t,e)]
```

Each token's captured top-eight weights already sum to approximately one. The vector is then normalized by that expert's accumulated weight mass, making it a weighted average in the original hidden-state coordinates. A same-split uniform mean was evaluated as a control. For the 734 layer/expert slots never selected in training, the code uses the context with that expert's highest saved native gate logit; those fallback vectors do not have contribution-weight evidence.

The evaluation reused the 384-prompt corpus and seed-23 topic-group split: 38 topic groups for fitting and 10 held out. Each of 40 layers has 17,792 training and 4,504 validation states. It reports exhaustive cosine, dot, and negative squared-L2 scoring for weighted and uniform means, then builds the deployment HNSW with one weighted vector per expert, squared-L2 keys, and 128 of 256 expert vectors retrieved per query. Per-layer softmax temperature is fitted on the training routing weights.

## Aggregate held-out metrics

| Vector | Score | Top-8 recall | Exact set | Calibrated weight L1 | Score Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| Contribution-weighted mean | cosine | 0.5914 | 0.0186 | 0.9802 | 0.4568 |
| Contribution-weighted mean | dot | 0.3195 | 0.0045 | 1.4545 | 0.4317 |
| Contribution-weighted mean | euclidean | 0.5961 | 0.0192 | 0.9739 | 0.4235 |
| Uniform mean | cosine | 0.5880 | 0.0210 | 1.0060 | 0.4617 |
| Uniform mean | dot | 0.3084 | 0.0060 | 1.4879 | 0.4372 |
| Uniform mean | euclidean | 0.5928 | 0.0206 | 0.9990 | 0.4301 |

The ANN router's weighted-L2 result was recall **0.5960**, exact set **0.0192**, weight L1 **0.9740**, and native top-eight candidate support **0.8989**. HNSW versus exhaustive weighted L2 changed mean recall by -0.000162, exact-set rate by +0.000006, and weight L1 by +0.000062.

Compared with the unweighted mean under exhaustive L2, contribution weighting changed recall by **+0.0034**, exact-set rate by **-0.0014**, and weight L1 by **-0.0251**. It improved L2 recall in 30/40 layers and weight L1 in 40/40, but improved exact-set rate in only 15/40.

Against the previous full-run HDBSCAN/PCA-384 ANN router:

| Metric | Weighted mean, 2,048D | HDBSCAN/PCA-384 ANN | Change |
| --- | ---: | ---: | ---: |
| Top-8 recall | 0.5960 | 0.6938 | -0.0978 |
| Exact expert set | 0.0192 | 0.2404 | -0.2212 |
| Routing-weight L1 | 0.9740 | 0.8272 | +0.1468 |
| Native-set candidate support | 0.8989 | 0.8307 | +0.0682 |
| Number of vectors | 10,240 | 80,509 | 70,269 fewer |

The previous comparison has different representation and search dimensions, so it tests the overall router redesign, not the isolated effect of contribution weighting. The uniform-mean control above is the clean comparison for that effect.

### Per-layer results

Layer numbering is zero-based. “Weighted exact L2” and “Uniform exact L2” use exhaustive scores over all 256 experts; weighted ANN retrieves 128 candidates through HNSW.

| Layer | Weighted ANN recall | Exact set | Weight L1 | Native-set support | Weighted exact L2 recall | Uniform exact L2 recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7388 | 0.0861 | 0.7516 | 0.9547 | 0.7386 | 0.7379 |
| 1 | 0.6258 | 0.0300 | 0.9311 | 0.8559 | 0.6259 | 0.6301 |
| 2 | 0.5447 | 0.0044 | 1.0949 | 0.7436 | 0.5448 | 0.5628 |
| 3 | 0.5793 | 0.0058 | 1.0341 | 0.8686 | 0.5793 | 0.5721 |
| 4 | 0.6429 | 0.0273 | 0.9540 | 0.9232 | 0.6433 | 0.6418 |
| 5 | 0.5834 | 0.0131 | 1.0101 | 0.8734 | 0.5836 | 0.5751 |
| 6 | 0.6037 | 0.0317 | 0.9672 | 0.9500 | 0.6039 | 0.5957 |
| 7 | 0.6007 | 0.0151 | 0.9716 | 0.8570 | 0.6011 | 0.5996 |
| 8 | 0.5649 | 0.0080 | 1.0299 | 0.8874 | 0.5651 | 0.5618 |
| 9 | 0.5761 | 0.0089 | 1.0425 | 0.9292 | 0.5761 | 0.5767 |
| 10 | 0.6176 | 0.0280 | 0.9264 | 0.9483 | 0.6178 | 0.6122 |
| 11 | 0.6320 | 0.0242 | 0.9184 | 0.9405 | 0.6320 | 0.6236 |
| 12 | 0.6020 | 0.0246 | 0.9913 | 0.9407 | 0.6024 | 0.6007 |
| 13 | 0.5389 | 0.0198 | 1.0884 | 0.9139 | 0.5390 | 0.5342 |
| 14 | 0.5866 | 0.0342 | 0.9699 | 0.8694 | 0.5866 | 0.5859 |
| 15 | 0.5660 | 0.0195 | 1.0061 | 0.8346 | 0.5661 | 0.5665 |
| 16 | 0.5981 | 0.0246 | 0.9911 | 0.9025 | 0.5982 | 0.5848 |
| 17 | 0.5674 | 0.0242 | 1.0336 | 0.8435 | 0.5678 | 0.5630 |
| 18 | 0.5566 | 0.0013 | 1.0319 | 0.8690 | 0.5569 | 0.5498 |
| 19 | 0.5820 | 0.0060 | 0.9738 | 0.8477 | 0.5820 | 0.5774 |
| 20 | 0.5308 | 0.0182 | 1.0760 | 0.8111 | 0.5309 | 0.5352 |
| 21 | 0.5491 | 0.0160 | 1.1241 | 0.8803 | 0.5493 | 0.5514 |
| 22 | 0.5902 | 0.0184 | 0.9689 | 0.9376 | 0.5907 | 0.5764 |
| 23 | 0.6119 | 0.0313 | 0.9301 | 0.9039 | 0.6121 | 0.6123 |
| 24 | 0.5606 | 0.0022 | 1.0826 | 0.8550 | 0.5609 | 0.5545 |
| 25 | 0.5379 | 0.0131 | 1.1199 | 0.8559 | 0.5385 | 0.5287 |
| 26 | 0.5606 | 0.0031 | 1.0265 | 0.8657 | 0.5606 | 0.5460 |
| 27 | 0.5699 | 0.0000 | 0.9887 | 0.8590 | 0.5697 | 0.5632 |
| 28 | 0.5660 | 0.0218 | 1.0882 | 0.8772 | 0.5661 | 0.5559 |
| 29 | 0.5788 | 0.0295 | 0.9968 | 0.9139 | 0.5787 | 0.5702 |
| 30 | 0.6050 | 0.0095 | 0.9353 | 0.9139 | 0.6052 | 0.5994 |
| 31 | 0.5462 | 0.0047 | 1.0344 | 0.9232 | 0.5464 | 0.5372 |
| 32 | 0.6488 | 0.0209 | 0.8508 | 0.9720 | 0.6491 | 0.6488 |
| 33 | 0.6366 | 0.0211 | 0.9233 | 0.9194 | 0.6370 | 0.6329 |
| 34 | 0.6795 | 0.0195 | 0.7904 | 0.9691 | 0.6794 | 0.6848 |
| 35 | 0.6032 | 0.0206 | 0.9054 | 0.9398 | 0.6033 | 0.6019 |
| 36 | 0.6254 | 0.0158 | 0.8643 | 0.9276 | 0.6257 | 0.6307 |
| 37 | 0.6287 | 0.0144 | 0.8363 | 0.9418 | 0.6291 | 0.6256 |
| 38 | 0.6049 | 0.0038 | 0.9726 | 0.9729 | 0.6052 | 0.6078 |
| 39 | 0.6978 | 0.0468 | 0.7268 | 0.9643 | 0.6979 | 0.6955 |

Across layers, ANN recall ranged from 0.5308 (layer 20) to 0.7388 (layer 0). Exact-set rate ranged from 0.0000 to 0.0861; weight L1 ranged from 0.7268 to 1.1241.

## Live generation and routing latency

Both generation checks used the full conditional-generation INT4 overlay, this router on all 40 layers, the same prompt, greedy decoding, and a 48-token maximum.

| Run | Generated | Time | Result |
| --- | ---: | ---: | --- |
| Default stopping | 48 tokens | 98.67 s | Repeated a confused description of PCA and the router; no useful explanation. |
| Minimum 16 tokens | 48 tokens | 90.77 s | Essentially the same repetitive text; the minimum length did not help. |

Router-only timing used 64 held-out hidden states per layer, three paired trials, and a CUDA synchronization after every layer decision. It excludes expert MLPs and the rest of the model.

| Router | Median per layer decision | Median across 40 layers/token |
| --- | ---: | ---: |
| Quantized native GPU gate | 731 microseconds | 29.23 ms |
| Weighted-vector HNSW (CPU search) | 2,977 microseconds | 119.07 ms |

The new route took **4.08×** as long as the native gate in this measurement (+89.84 ms per token across 40 layers). On 640 parity samples, the two routers selected the same unordered top-eight set 7 times (1.09%) and never matched the same ordering.

## Interpretation

Contribution weighting carries a useful signal: compared with a plain mean, it improves recall and weight error modestly on most layers. But the result is still a poor replica of the native router, and the live responses confirm that routing quality is inadequate for generation.

The weights answer “how large was this expert's share after the native router had already selected its top eight?” They do not provide a standalone score for the expert against all 256 alternatives or a counterfactual measure of what the expert contributes to the model's loss. A weighted mean summarizes contexts where the expert was selected; it does not learn the gate's decision boundary. The near equality of exhaustive and HNSW scores shows that adding a faster approximate index alone will not fix this quality gap.

The prototype bank is compact relative to the earlier multi-cluster bank, but the measured CPU HNSW path is slower than the small native GPU gate at this scale. This experiment establishes the weighted-mean baseline; it does not show a speed or answer-quality win.

## Reproduction and artifacts

```powershell
& .\.venv\Scripts\python.exe experiments\build_weighted_expert_vector_router.py `
  --trace-dir traces\expert_profile_qwen36 `
  --output-dir results\weighted_expert_vectors_qwen36_2048_20261009 `
  --device cuda:0

& .\.venv\Scripts\python.exe experiments\infer_qwen_moe_int4_ann.py `
  --router-dir results\weighted_expert_vectors_qwen36_2048_20261009 `
  --prompt "Explain how PCA can help a nearest-neighbor router using a short example." `
  --max-new-tokens 48
```

Detailed per-layer metrics and vectors are in [`vector_metrics.json`](../../results/weighted_expert_vectors_qwen36_2048_20261009/vector_metrics.json), the runtime bank is [`hidden2048_expert_bank.npz`](../../results/weighted_expert_vectors_qwen36_2048_20261009/hidden2048_expert_bank.npz), and HNSW indexes are in `ann_db/`. The router-only timing is in [`weighted_expert_vectors_router_only_benchmark_20261009.json`](../../results/weighted_expert_vectors_router_only_benchmark_20261009.json). Generation logs are in `artifacts/runlogs/weighted_vectors_generation_default_20261009.out.log` and `artifacts/runlogs/weighted_vectors_generation_min16_20261009.out.log`.

The build took 359.4 seconds. The compressed 2,048D vector bank is 71.9 MiB and the 40 HNSW files total 42.7 MiB. The evaluation is one checkpoint revision and one topic split, with no repeated-seed confidence interval. The traces are prompt-profile hidden states, not states from free-running generation.
