# Router design diagnosis and a gate-preserving ANN control

**Checkpoint:** Qwen/Qwen3.6-35B-A3B, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. This is the selected post-trained checkpoint.  
**Environment:** Windows, RTX 5090 32 GB, RTX 3090 24 GB, Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.19.0. The new generation used the RTX 5090.  
**Experiment completion:** 2026-10-09 01:17:20 America/New_York. The filename timestamp comes from the final generation log artifact; the process did not record a wall-clock completion timestamp.

## Finding

The original context-prototype router substitutes a different decision rule for Qwen's trained gate. It describes where an expert was used in the profiling corpus; it does not recover the learned rule for choosing that expert or its contribution. PCA also removes some gate-sensitive directions, and the profiling corpus omits chat formatting and assistant decoding. Replacing all 40 gates lets these errors accumulate.

This diagnosis produced a working alternative control: use the learned gate weight vectors as routing anchors, retain their rowspace instead of high-variance activation directions, and search with a metric that preserves gate ordering. Across all 180,160 held-out layer-token observations, its ANN candidates included every native top-8 expert. Final recall was 0.999988216, exact-set rate 0.999905638, and weight L1 2.071320e-5 relative to the same quantized native gates. A live 48-token generation produced ordinary prompt-related prose rather than repeated thinking markers. The short generation is a smoke test, not a broad quality evaluation.

## Why the original intuition breaks

### Usage centroids do not recover a trained gate

Qwen's gate is a bias-free linear function:

```text
native score for expert e:       s_e(h) = w_e · h
selected experts:               top8(s(h))
mixture weights:                softmax on the selected eight scores
```

The context bank instead computes:

```text
z = PCA(h - mean)
prototype score for expert e:   max_j -||z - c_ej||²
```

After dropping the query-wide term -||z||², a centroid's score is 2 z·c_ej - ||c_ej||². Its direction and norm therefore become a classifier direction and an expert-specific bias. HDBSCAN fits activation density; it does not train those directions and biases to match w_e·h. Multiple prototypes per expert also create unequal opportunities to obtain a close match. Temperature scales the scores and changes mixture sharpness, but a positive scalar cannot repair the expert ordering.

A simple example illustrates the distinction. Suppose one expert's contexts have centroid (1,0), another's centroid is (4,0), and a query moves from (1.5,0) to (3,0). Nearest-centroid selection switches experts. A native gate with weight vectors (2,0) and (1,0) keeps the first expert at both queries because all logits scale together. The two routers encode different functions even before ANN approximation or compression.

The original held-out recall of 0.693807 means about 2.45 of eight expert selections differ per layer on native trace states. The exact-set rate of 0.240397 corresponds to 30.4 changed layer sets per token on average. These arithmetic averages require no independence assumption. They do not quantify states reached after the replacement router has already changed earlier layers. The mean routing-weight L1 of 0.827220 represents 0.413610 total variation between the mixtures; expert overlap alone understates the computation change.

### PCA optimizes activation variance, not gate fidelity

PCA-384 retained 82.6964–90.2430% of activation variance. A component with low activation variance can still carry a large discriminative gate weight, so that number is not a routing guarantee.

The new diagnostic used the same group-held-out traces and the original gate matrices. It compared captured native expert IDs with four reconstructions. For PCA, the mean contribution was retained explicitly:

```text
z = (h - mean) Cᵀ
reconstructed logits = z (W Cᵀ)ᵀ + mean Wᵀ
```

For the gate-aware condition, QR factorization supplied an orthonormal basis Q spanning the 256 gate weight vectors, and logits were reconstructed as (h Q)(W Q)ᵀ. The basis has at most 256 dimensions and preserves W exactly in real arithmetic. The maximum measured relative weight projection residual was 4.150e-7.

| Diagnostic | Reference | Mean top-8 recall |
|---|---|---:|
| Original HDBSCAN context centroids + PCA-384 + HNSW | Captured BF16 native IDs | 0.693807 |
| Original learned gate reconstructed through activation PCA-384 | Captured BF16 native IDs | 0.828467 |
| Original learned gate reconstructed through its 256D rowspace | Captured BF16 native IDs | 0.999639 |
| Original gate recomputed directly from saved hidden states | Captured BF16 native IDs | 0.999629 |
| New gate-anchor L2 ANN + candidate reranking | Quantized native gates applied to the same hidden states | 0.999988216 |

The rowspace result matches direct native reconstruction to numerical precision. The gap from 0.828467 through PCA to 0.693807 with contextual centroids also shows that changing the scoring rule adds substantial error beyond compression. These conditions use different scoring mechanisms; they are not a complete causal ablation of all clustering choices.

Top-8 recall is the mean expert-set intersection size divided by eight. Exact-set rate is the fraction with an identical eight-expert set. Ordered-ID rate also requires the same eight positions. Weight L1 sums absolute differences between normalized mixtures in a dense 256-expert vector. Aggregate values are arithmetic means over 40 layers; each complete validation layer has 4,504 positions. There was one split, with no repeated-seed uncertainty estimate.

### The bank does not cover deployment states

The profile contained 384 prompts from 48 topics: original prompts plus seven repeated instruction frames. Topic-group splitting with seed 23 and target training fraction 0.8 held out ten groups and kept 38 in training. Held-out topics were 004, 005, 017, 031, 032, 041, 042, 043, 044, and 047. Each layer had 17,792 training and 4,504 validation positions.

The original builder fitted per-layer randomized PCA-384 on training activations, then clustered the projected contexts selected for each native top-8 expert. HDBSCAN used Euclidean distance, EOM selection, minimum cluster size 32, and minimum samples 5. Cluster means formed prototypes; small noise sets were assigned to nearby clusters, larger noise sets were pooled into one prototype, and insufficient/no-cluster support produced marked fallbacks. The bank contained 80,509 prototypes across 40 USearch squared-L2 HNSW indexes. Original index settings were connectivity 32, expansion-add 128, expansion-search 128, and 128 retrieved prototypes per query. The replacement gate-anchor condition below has separate settings and a separate bank.

Profiling used raw prompt tokenization, BF16 native routes, prefill, and no cache. Deployment uses a chat template, assistant boundaries, INT4 hidden states, and cached autoregressive decoding. No assistant continuations or states produced after prior custom-router substitutions were profiled.

Coverage was also uneven across 10,240 layer-expert slots:

| Support measure | Value |
|---|---:|
| Zero observed native top-8 support | 734 slots |
| Fewer than 32 selected contexts | 2,538 slots |
| Fallback prototypes | 4,574 slots (44.66797%) |
| Median selected-context count | 169 |
| Largest selected-context count | 17,639 of 17,792 training tokens |

A fallback for an unsupported expert uses its highest-logit context even when that expert was never selected. That makes the ID retrievable, but does not establish that it should be used there. Pooling disconnected HDBSCAN noise into one centroid can also place a prototype between useful modes.

At layer 0, exact byte comparison of saved FP16 hidden-state rows found that 1,658 of 4,504 validation rows (36.8117%) duplicated a training row. Shared instruction prefixes create these repeated states despite the correct topic split. This was measured at layer 0 only. It makes the aggregate held-out score less representative of genuinely new contexts; it does not establish topic-content leakage.

### Calibration and layer replacement amplify the error

The fitted scalar temperature minimizes cross-entropy only within the teacher's eight selected experts, excluding tokens where that full set is absent from retrieved candidates. Inference applies it to the set the ANN actually selects. Thus calibration is conditioned on a better expert set than the deployed router receives.

All 40 gates were replaced together. Each wrong expert mixture changes the hidden input to the next layer, and each generated token changes the next prefix. Native-state validation does not measure this accumulating state shift.

## What the implementation audit ruled out

The runtime and builder produced matching contextual-prototype scores on 128 held-out rows each from layers 0, 20, and 39. An exhaustive search over every context prototype barely improved native recall:

| Layer | Sample size | HNSW recall | Exhaustive recall | HNSW/exact set overlap | Exhaustive weight L1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.813477 | 0.823242 | 0.982422 | 0.658756 |
| 20 | 128 | 0.602539 | 0.611328 | 0.985352 | 0.943589 |
| 39 | 128 | 0.768555 | 0.775391 | 0.980469 | 0.608228 |

This suggests prototype geometry and calibration dominate the original search approximation on these sampled states. The result is sampled evidence, not an exhaustive HNSW audit of every trace.

The current quantized expert calculation follows the native gate/up activation, down projection, and weighted accumulation. Native-gate INT4 generation succeeds, so the recorded ANN failure is not explained by an ANN-specific quantization defect. That does not claim BF16 and INT4 are identical models.

A latent bug was fixed: multiplying absent -infinity candidate scores by a zero temperature would create NaNs. The saved bank's temperatures were all positive, from about 0.0001400 to 0.0012864, so this bug did not cause its recorded failure. The runtime now selects finite unscaled scores first, then applies temperature and checks weights. Six focused tests passed, covering zero temperature with absent candidates, positive-temperature parity, invalid temperatures, too few finite candidates, nonfinite scores, and overflow.

The recorded bank was built by clustering contexts after PCA. The requested sequence was HDBSCAN in 2048D followed by prototype compression. Those neighborhoods differ. The builder now defaults to clustering full 2048D contexts and projecting their means afterward; a projected-space option preserves the earlier method for reproduction. The full-space variant has not been run in this diagnosis and would not, by itself, make centroid distances equivalent to the trained gate.

## Gate-anchor control: construction and evidence

### Preserve the trained information

For each of the 40 layers, the control dequantized its native gate matrix from the same groupwise INT4 overlay used by inference. Each matrix has 256 rows and 2048 columns. QR produced Q with 256 orthonormal columns:

```text
query coordinates:       z = Qᵀ h
expert anchor:           a_e = Qᵀ w_e
gate score:              a_e · z = w_e · h
```

The untrained activation PCA objective is replaced by a subspace containing every learned gate direction. This equality is independent of whether h comes from a prompt, a chat boundary, or assistant decoding.

The initial direct inner-product HNSW condition retained all native top-8 candidates on an average fraction of 0.979199 of sampled contexts (256 contexts per layer). Inner product is the correct scoring function, but that approximate search configuration still missed candidates. The L2 control therefore used an equal-norm lift:

```text
R² >= max_e ||a_e||²
index key for expert e:  [a_e, sqrt(R² - ||a_e||²), zeros...]
query:                   [z,   0,                  zeros...]
squared distance:        ||z||² + R² - 2 a_e·z
```

The keys have 384 coordinates: 256 gate coordinates, one lifting coordinate, and 127 zeros. Because the first two distance terms are common to experts, nearest L2 keys preserve the original maximum-dot-product ordering. HNSW used float32 keys, connectivity 32, expansion 256, and a 32-expert shortlist. The original gate vectors were retained to rerank those 32 candidates and compute their native logits; this control is not a pure centroid-distance mixture.

### Preserve finite-precision selection

FP32 candidate accumulation is rounded to BF16. The rerank logits are then scattered into the original 256 expert positions before softmax and top-k. This matters because BF16 gate logits often tie. Top-k over a reordered 32-candidate array changed tie choices even when all teacher candidates and their exact logits were present. On a targeted 256-context layer-20 test, restoring the 256 expert positions changed exact-set rate from 0.847656 to 1.0 and reduced unrounded weight L1 to 4.89e-8.

The final deployed runtime was validated on every held-out position, 4,504 per layer, and compared with the same quantized native gate weights. Native and candidate weights were compared in their actual BF16 output dtype.

| Metric | Mean over 40 layers / 180,160 observations |
|---|---:|
| All native top-8 experts included among 32 ANN candidates | 1.000000000 |
| Final top-8 recall | 0.999988216 |
| Exact eight-expert set | 0.999905638 |
| Ordered expert IDs | 0.999672511 |
| Routing-weight L1 | 2.071320e-5 |

The tiny residual differences occur around finite-precision scores and ties. The equivalence is mathematical; HNSW candidate recall and floating-point behavior are empirical constraints. The validation uses raw BF16 native-trace hidden states with quantized gate matrices, not a corpus of quantized free-running states.

### Live generation

The control used the full conditional-generation INT4 model, the same PCA/nearest-neighbor explanation request used in the earlier smoke tests, the chat template with thinking disabled, greedy decoding, and a 48-token cap. It installed gate-anchor routing on all 40 layers and produced:

> # PCA (Principal Component Analysis) for a Nearest-Neighbor Router
>
> ## Conceptual Overview
>
> A “nearest-neighbor router” typically refers to a system that:
> 1. **Ranks/weights** the importance of data points

Generation reached 48 tokens in 96.01 seconds. The response is ordinary prose related to the request, but it is capped before a complete explanation or example. It does not establish accuracy, completeness, or broad quality. Earlier runs with the context-centroid router returned only a thinking marker after two tokens, or repeated thinking markers through a forced 48-token cap. The native-gate INT4 control also produced normal prose at the same 48-token cap, in 85.72 seconds. These separate runs are not a controlled throughput benchmark, and no speed improvement is demonstrated.

## Practical role for the Proteus idea

### The gate is currently a small target

A native per-layer gate costs 256×2048 = 524,288 multiply-adds and 1 MiB of BF16 weights. All 40 gate matrices occupy 40 MiB. The original PCA projection alone costs 384×2048 = 786,432 multiply-adds per layer, 1.5 times the gate's arithmetic, before HNSW lookup and a GPU-to-CPU synchronization. The PCA matrices occupy 120 MiB float32; the 80,509 raw prototype vectors occupy about 117.9 MiB before index overhead.

Consequently, ANN replacement at only 256 native experts is not yet targeting the dominant compute or memory cost. The learned-anchor version establishes routing fidelity and provides a useful control; its projection, candidate reranking, and CPU search add work and retain original candidate weight vectors. It has not been optimized for speed.

### Retain HDBSCAN where density information is useful

A practical use is predicting future experts for GPU prefetch and cache residency while the inexpensive native gate selects the current token's experts. A wrong prefetch prediction affects a cache hit or latency rather than changing the model computation. This needs sequential native-route data, cache-hit/coverage measurements, and a hardware latency experiment; no such performance outcome has been measured here.

Another use is selecting trained specialist adapters or packages once per request or text segment. That is a coarser decision with fewer CPU/GPU synchronization points. The specialist's routing key and its expected hidden representation still need compatibility training and output-quality evaluation. Qwen's individual per-layer feed-forward experts are coadapted components, not demonstrated independent topic models that can be freely rearranged.

### If contextual prototypes must decide current-token routing

The context prototype router needs supervision against the native computation:

1. Retain gate-sensitive coordinates, with the native gate rowspace as an exact initialization.
2. Optimize prototype keys, per-expert offsets, and mixture weights against full teacher logits and MoE outputs. Do not calibrate only on the oracle top-8 subset.
3. Capture diverse chat inputs, teacher assistant continuations, cached decode, and states reached under the replacement router. Score shared boilerplate separately and hold out whole conversations/documents.
4. Treat unsupported experts explicitly, using the native gate for those cases until data and training support replacement.
5. Replace a few layers first and broaden only after teacher-forced next-token distributions and free-running response quality stay close. Expert-ID recall is an intermediate diagnostic; answer quality is the acceptance criterion.

More context data can improve coverage, but it cannot repair the scoring-function mismatch by itself. The learned anchors demonstrate a low-risk route to preserving existing behavior; contextual clusters need an objective that learns expert utility rather than assuming nearest usage means highest contribution.

## Complete layer results

“Context” and “PCA gate” are compared with captured BF16 native expert IDs. “Rowspace gate” uses the same reference and matches direct native reconstruction precision. “Anchor” is compared with the quantized native gates on the same 4,504 held-out hidden states per layer. These references differ deliberately and should not be treated as a precision-format ablation. PCA variance is retained activation variance; it is not gate fidelity.

| Layer | PCA variance | Context recall | PCA gate recall | Rowspace gate recall | Anchor recall | Anchor exact set | Anchor weight L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9024 | 0.8150 | 0.9179 | 1.000000 | 1.000000 | 1.000000 | 4.120e-6 |
| 1 | 0.8654 | 0.7369 | 0.8646 | 1.000000 | 1.000000 | 1.000000 | 1.100e-5 |
| 2 | 0.8495 | 0.6892 | 0.8240 | 1.000000 | 0.999972 | 0.999778 | 4.027e-5 |
| 3 | 0.8351 | 0.6942 | 0.7970 | 1.000000 | 0.999945 | 0.999556 | 6.879e-5 |
| 4 | 0.8369 | 0.7638 | 0.8544 | 0.999972 | 1.000000 | 1.000000 | 9.811e-6 |
| 5 | 0.8352 | 0.6845 | 0.8206 | 1.000000 | 1.000000 | 1.000000 | 1.122e-5 |
| 6 | 0.8427 | 0.6952 | 0.8324 | 1.000000 | 1.000000 | 1.000000 | 7.914e-6 |
| 7 | 0.8344 | 0.6808 | 0.8111 | 1.000000 | 1.000000 | 1.000000 | 1.843e-6 |
| 8 | 0.8318 | 0.6825 | 0.7909 | 1.000000 | 0.999972 | 0.999778 | 3.637e-5 |
| 9 | 0.8370 | 0.6891 | 0.8187 | 1.000000 | 1.000000 | 1.000000 | 2.873e-6 |
| 10 | 0.8445 | 0.7193 | 0.8519 | 1.000000 | 0.999972 | 0.999778 | 1.420e-5 |
| 11 | 0.8336 | 0.7273 | 0.8508 | 1.000000 | 1.000000 | 1.000000 | 7.643e-6 |
| 12 | 0.8334 | 0.7031 | 0.8114 | 1.000000 | 0.999972 | 0.999778 | 3.767e-5 |
| 13 | 0.8316 | 0.6783 | 0.8052 | 1.000000 | 0.999972 | 0.999778 | 3.193e-5 |
| 14 | 0.8382 | 0.6641 | 0.8006 | 1.000000 | 1.000000 | 1.000000 | 8.131e-6 |
| 15 | 0.8327 | 0.6625 | 0.7966 | 1.000000 | 1.000000 | 1.000000 | 7.372e-6 |
| 16 | 0.8313 | 0.6907 | 0.8170 | 0.998807 | 1.000000 | 1.000000 | 1.035e-5 |
| 17 | 0.8317 | 0.6774 | 0.8055 | 0.998945 | 1.000000 | 1.000000 | 8.348e-6 |
| 18 | 0.8338 | 0.6667 | 0.8093 | 0.998668 | 1.000000 | 1.000000 | 7.806e-6 |
| 19 | 0.8285 | 0.6438 | 0.7974 | 0.998751 | 1.000000 | 1.000000 | 6.288e-6 |
| 20 | 0.8270 | 0.6226 | 0.7729 | 0.998751 | 1.000000 | 1.000000 | 5.637e-6 |
| 21 | 0.8322 | 0.6642 | 0.8123 | 0.998973 | 1.000000 | 1.000000 | 5.312e-6 |
| 22 | 0.8373 | 0.6968 | 0.8462 | 0.999029 | 1.000000 | 1.000000 | 5.583e-6 |
| 23 | 0.8342 | 0.6819 | 0.8352 | 0.998668 | 1.000000 | 1.000000 | 6.830e-6 |
| 24 | 0.8297 | 0.6632 | 0.8066 | 0.998945 | 0.999972 | 0.999778 | 4.775e-5 |
| 25 | 0.8284 | 0.6654 | 0.8037 | 0.999056 | 1.000000 | 1.000000 | 7.480e-6 |
| 26 | 0.8381 | 0.6592 | 0.8099 | 0.998779 | 0.999945 | 0.999556 | 8.180e-5 |
| 27 | 0.8376 | 0.6489 | 0.8045 | 0.998224 | 0.999972 | 0.999778 | 1.924e-5 |
| 28 | 0.8436 | 0.6851 | 0.8330 | 1.000000 | 0.999972 | 0.999778 | 3.393e-5 |
| 29 | 0.8515 | 0.6790 | 0.8323 | 1.000000 | 1.000000 | 1.000000 | 2.927e-6 |
| 30 | 0.8541 | 0.6711 | 0.8348 | 1.000000 | 0.999972 | 0.999778 | 4.033e-5 |
| 31 | 0.8432 | 0.6667 | 0.8328 | 1.000000 | 1.000000 | 1.000000 | 7.697e-6 |
| 32 | 0.8519 | 0.7326 | 0.8666 | 1.000000 | 1.000000 | 1.000000 | 3.523e-6 |
| 33 | 0.8517 | 0.7456 | 0.8731 | 1.000000 | 0.999972 | 0.999778 | 3.881e-5 |
| 34 | 0.8498 | 0.7693 | 0.8831 | 1.000000 | 1.000000 | 1.000000 | 8.781e-6 |
| 35 | 0.8378 | 0.6644 | 0.8109 | 1.000000 | 1.000000 | 1.000000 | 1.377e-5 |
| 36 | 0.8398 | 0.6877 | 0.8232 | 1.000000 | 0.999972 | 0.999778 | 4.868e-5 |
| 37 | 0.8428 | 0.7121 | 0.8351 | 1.000000 | 0.999945 | 0.999556 | 1.009e-4 |
| 38 | 0.8560 | 0.7116 | 0.8555 | 1.000000 | 1.000000 | 1.000000 | 7.155e-6 |
| 39 | 0.8591 | 0.7603 | 0.8897 | 1.000000 | 1.000000 | 1.000000 | 8.402e-6 |

## Limits

This is one checkpoint revision and one topic-group split. Exact contextual-prototype search was tested on only 128 contexts for three layers. The projection and final gate-anchor validation covered all held-out native trace states. The live generation covered one prompt at 48 tokens and no image input. No broad response-quality benchmark, expert-prefetch experiment, specialist-module compatibility experiment, or throughput advantage has been established. The modified numerical guards improve robustness but do not explain away the original contextual router's measured failure.
