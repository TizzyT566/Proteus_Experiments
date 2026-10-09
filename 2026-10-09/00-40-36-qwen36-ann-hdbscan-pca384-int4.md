# Full Qwen3.6 HDBSCAN–PCA-384 ANN Router and INT4 Inference

**Run date:** 2026-10-08 to 2026-10-09, America/New_York  
**Report timestamp:** 2026-10-09 00:40:36 America/New_York. The filename time comes from the final inference output artifact because the process did not record its own completion timestamp.  
**Outcome:** The full Qwen conditional-generation model loads with the packed INT4 weights and can generate with its native gates. The ANN router builds and installs across all 40 MoE layers, but the tested ANN-routed generations are not usable.

## Objective and checkpoint

This experiment exercised the requested end-to-end path: expand a broad text corpus, profile all expert gates, cluster routed contexts per expert, compress prototypes to 384 dimensions, index them with HNSW, quantize the selected Qwen checkpoint to groupwise INT4, and generate using the ANN router.

The checkpoint was `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. This is the post-trained public checkpoint selected for the run, not the separate Base checkpoint named in earlier project notes. Router profiling used its text causal language model; quantization and generation used its `Qwen3_5MoeForConditionalGeneration` class, including the vision tower. The inference checks in this report used text prompts only. The checkpoint's auxiliary MTP training tensors are not part of the conditional-generation class used for inference.

## Corpus and profiling

The source set had 48 broad topic groups. Eight deterministic prompt forms per group produced 384 prompts. Ten whole topic groups were held out for validation, leaving 38 groups for training. The held-out groups were topics 004, 005, 017, 031, 032, 041, 042, 043, 044, and 047. Splitting by topic group kept prompts from the same topic out of both partitions.

The unquantized text model has 40 MoE layers, 256 experts per layer, and a 2,048-dimensional hidden state. Gate traces covered all layers. Each layer contained 22,296 traced token positions: 17,792 training and 4,504 validation positions. Across layers this is 711,680 training and 180,160 validation layer-token observations.

For each layer, PCA was fitted on the training hidden states and projected the 2,048-dimensional inputs to 384 dimensions. PCA retained between 82.6964% and 90.2430% of variance by layer.

For each expert, training contexts routed to it in the native top-8 gate were clustered independently with HDBSCAN using Euclidean distance, `min_cluster_size=32`, `min_samples=5`, and EOM selection. A prototype was the mean of a cluster's contexts. Noise smaller than 32 contexts was attached to its nearest cluster center; a noise set of at least 32 contexts became its own prototype. When no density cluster could be formed, the builder kept a marked fallback: the mean routed context, or, for an expert never selected in training, the projected context with that expert's highest native gate score. This made all 256 expert IDs addressable in every layer.

The resulting bank had 80,509 prototypes total (1,884–2,160 per layer). Of the 10,240 layer-expert slots, 5,666 had at least one HDBSCAN density cluster; 4,574 slots used one fallback prototype. Thus 44.67% of layer-expert slots were represented by fallback rather than a density cluster. The fallback preserves addressability, but it is weaker evidence about an expert's context distribution.

## ANN scoring and validation

Each layer's projected prototypes were indexed in a USearch HNSW index with squared-L2 distance. Validation and inference queried 128 nearest prototypes by default. Prototype labels were mapped to expert IDs; each expert's score was the negative squared distance of its nearest returned prototype. A per-layer scalar temperature was selected on training tokens to minimize cross-entropy against native top-8 routing weights. The ANN then selected eight experts and normalized their scores into weights.

Metrics below are computed on held-out validation tokens. `Recall@8` is the mean intersection size between the ANN and native expert sets divided by eight. `Exact set` is the fraction of tokens whose full eight-expert sets match. `Weight L1` is the mean L1 distance between the dense 256-way vectors of native and ANN top-8 weights. `Support` is the fraction of tokens for which all eight native experts appeared among the ANN candidates. Aggregate values are arithmetic means across the 40 layers; there was one topic-group split and no repeated-seed uncertainty estimate.

| Metric | Mean across 40 layers |
|---|---:|
| Native top-8 recall | 0.693807 |
| Exact top-8 set rate | 0.240397 |
| Top-8 weight L1 | 0.827220 |
| Native-set support among ANN candidates | 0.830739 |

Layer-level results:

| Layer | Train tokens | Validation tokens | Prototypes | Fallbacks | PCA variance | Recall@8 | Exact set | Weight L1 | Support |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17,792 | 4,504 | 2,037 | 88 | 0.902430 | 0.815026 | 0.410968 | 0.675175 | 0.844805 |
| 1 | 17,792 | 4,504 | 1,950 | 82 | 0.865429 | 0.736873 | 0.280195 | 0.781350 | 0.863899 |
| 2 | 17,792 | 4,504 | 1,921 | 78 | 0.849529 | 0.689165 | 0.291297 | 0.914531 | 0.754440 |
| 3 | 17,792 | 4,504 | 1,884 | 108 | 0.835076 | 0.694161 | 0.257771 | 0.868181 | 0.781750 |
| 4 | 17,792 | 4,504 | 1,894 | 128 | 0.836949 | 0.763821 | 0.321936 | 0.776826 | 0.912744 |
| 5 | 17,792 | 4,504 | 1,984 | 124 | 0.835176 | 0.684475 | 0.200710 | 0.877035 | 0.810169 |
| 6 | 17,792 | 4,504 | 2,008 | 132 | 0.842666 | 0.695243 | 0.246004 | 0.837054 | 0.873446 |
| 7 | 17,792 | 4,504 | 1,990 | 117 | 0.834418 | 0.680784 | 0.220471 | 0.857569 | 0.779085 |
| 8 | 17,792 | 4,504 | 1,964 | 119 | 0.831814 | 0.682532 | 0.227131 | 0.835616 | 0.848357 |
| 9 | 17,792 | 4,504 | 1,985 | 121 | 0.836976 | 0.689082 | 0.212256 | 0.856394 | 0.876110 |
| 10 | 17,792 | 4,504 | 2,010 | 140 | 0.844497 | 0.719333 | 0.247780 | 0.784295 | 0.901421 |
| 11 | 17,792 | 4,504 | 1,985 | 126 | 0.833627 | 0.727326 | 0.276421 | 0.789494 | 0.916963 |
| 12 | 17,792 | 4,504 | 1,922 | 127 | 0.833426 | 0.703125 | 0.268206 | 0.838706 | 0.811723 |
| 13 | 17,792 | 4,504 | 1,983 | 124 | 0.831599 | 0.678341 | 0.235346 | 0.853036 | 0.849911 |
| 14 | 17,792 | 4,504 | 2,039 | 100 | 0.838159 | 0.664104 | 0.195160 | 0.853136 | 0.720693 |
| 15 | 17,792 | 4,504 | 1,966 | 98 | 0.832742 | 0.662522 | 0.199822 | 0.852317 | 0.697824 |
| 16 | 17,792 | 4,504 | 2,002 | 127 | 0.831303 | 0.690719 | 0.251110 | 0.864842 | 0.861456 |
| 17 | 17,792 | 4,504 | 2,006 | 116 | 0.831720 | 0.677398 | 0.253330 | 0.852013 | 0.771980 |
| 18 | 17,792 | 4,504 | 2,012 | 112 | 0.833824 | 0.666685 | 0.232460 | 0.860570 | 0.765986 |
| 19 | 17,792 | 4,504 | 2,035 | 99 | 0.828507 | 0.643844 | 0.195160 | 0.880765 | 0.767540 |
| 20 | 17,792 | 4,504 | 1,982 | 96 | 0.826964 | 0.622641 | 0.184503 | 0.922418 | 0.765320 |
| 21 | 17,792 | 4,504 | 2,088 | 97 | 0.832162 | 0.664187 | 0.227353 | 0.945528 | 0.747558 |
| 22 | 17,792 | 4,504 | 1,984 | 109 | 0.837336 | 0.696825 | 0.240453 | 0.804738 | 0.897202 |
| 23 | 17,792 | 4,504 | 2,012 | 105 | 0.834162 | 0.681866 | 0.236456 | 0.848497 | 0.807726 |
| 24 | 17,792 | 4,504 | 1,987 | 107 | 0.829663 | 0.663244 | 0.231794 | 0.929545 | 0.773535 |
| 25 | 17,792 | 4,504 | 2,040 | 109 | 0.828391 | 0.665353 | 0.211590 | 0.906918 | 0.794849 |
| 26 | 17,792 | 4,504 | 2,007 | 105 | 0.838105 | 0.659220 | 0.253552 | 0.865840 | 0.836368 |
| 27 | 17,792 | 4,504 | 1,978 | 98 | 0.837632 | 0.648923 | 0.184725 | 0.874962 | 0.716252 |
| 28 | 17,792 | 4,504 | 2,045 | 119 | 0.843602 | 0.685141 | 0.252442 | 0.896075 | 0.825488 |
| 29 | 17,792 | 4,504 | 2,031 | 109 | 0.851514 | 0.678952 | 0.218250 | 0.835257 | 0.864343 |
| 30 | 17,792 | 4,504 | 2,118 | 110 | 0.854138 | 0.671070 | 0.218028 | 0.842907 | 0.824378 |
| 31 | 17,792 | 4,504 | 2,105 | 109 | 0.843243 | 0.666685 | 0.206705 | 0.829048 | 0.800400 |
| 32 | 17,792 | 4,504 | 2,029 | 125 | 0.851900 | 0.732627 | 0.242895 | 0.729787 | 0.938721 |
| 33 | 17,792 | 4,504 | 2,083 | 128 | 0.851743 | 0.745587 | 0.266874 | 0.756499 | 0.876110 |
| 34 | 17,792 | 4,504 | 2,049 | 138 | 0.849770 | 0.769261 | 0.264876 | 0.659574 | 0.911856 |
| 35 | 17,792 | 4,504 | 2,025 | 108 | 0.837759 | 0.664437 | 0.197602 | 0.818367 | 0.835702 |
| 36 | 17,792 | 4,504 | 2,020 | 114 | 0.839769 | 0.687667 | 0.207149 | 0.777322 | 0.884991 |
| 37 | 17,792 | 4,504 | 2,068 | 137 | 0.842794 | 0.712062 | 0.224689 | 0.706762 | 0.878108 |
| 38 | 17,792 | 4,504 | 2,160 | 136 | 0.855951 | 0.711645 | 0.245115 | 0.792832 | 0.930284 |
| 39 | 17,792 | 4,504 | 2,121 | 149 | 0.859087 | 0.760324 | 0.277309 | 0.637026 | 0.910080 |

Layer 0 had the highest recall (0.815026); layer 20 had the lowest (0.622641). Exact set accuracy ranged from 0.184503 to 0.410968. The metric gap and high weight L1 show that “all experts are indexed” did not translate into faithful gate behavior.

## INT4 overlay and inference setup

The conditional-generation model instantiated 35,107,181,936 parameters. Symmetric signed groupwise INT4 used two packed 4-bit values per byte and one FP16 scale per group of 128 weights. Dense layers, embeddings, convolutions, native gate weights, and the packed MoE expert weights were converted; packed expert tensors were grouped independently per expert so scales did not cross expert boundaries. The final overlay has 614 quantized modules and 1,228 packed tensors. It quantizes 35,106,598,912 weights; 583,024 bias and normalization parameters remain at source precision. The overlay occupies about 18.10 GB (16.86 GiB).

Quantization ran on the Windows host with CUDA available for weight conversion. Inference dispatch placed the model on the RTX 5090, using about 23 GB of its 32 GB VRAM. The RTX 3090 was not needed for the observed device map. At runtime, packed weights are dequantized to BF16/FP16 for ordinary PyTorch operations. This is a portable weight-only storage format, not a fused INT4 GEMM kernel. Transformers used reference PyTorch fallbacks for causal convolution and gated-delta operations; those warnings indicate slower kernels, not a failed model load.

## Generation controls

All runs used the same short text request, greedy decoding, a 48-token cap, and the final per-expert-grouped overlay.

| Router and setting | Result | Time |
|---|---|---:|
| Native top-8 gates | Produced a coherent, prompt-relevant explanation of PCA and nearest-neighbor routing; output was truncated by the 48-token cap. | 85.72 s |
| ANN router, default stopping | Stopped after 2 tokens and returned only a `<think>` marker. | 15.78 s |
| ANN router, minimum 16 generated tokens | Repeated `<think>` markers through the 48-token cap; no user-facing answer. | 96.07 s |

The native-gate control establishes that the packed conditional-generation model can load and generate on the Windows GPU. The ANN run installed routing on all 40 layers without runtime errors, but it did not produce a useful answer. Forcing a minimum length did not fix this. The validation recall and exact-set results are consistent with a router-quality problem; this small control set does not isolate whether errors originate from prototype density, distance scoring/calibration, HNSW candidate support, or their combined effect.

## Conclusion and limitations

The requested corpus, 40-layer profiling, HDBSCAN prototype bank, PCA-384 projections, HNSW indexes, full conditional-generation INT4 overlay, and inference runner are all built. The corrected overlay has a per-expert packing layout verified for all 256 expert rows. Native-gate Q4 generation works, and the ANN routing path executes, but the current prototype router is not suitable for generation quality: it has 0.693807 mean top-8 recall and 0.240397 exact-set rate, and both tested ANN generations failed to return a normal answer.

This was a single checkpoint revision, one corpus expansion, and one topic-group split. The selected checkpoint is post-trained rather than Base. The smoke test covered text generation only; image input was not evaluated. No fused INT4 kernel was used, no repeated-seed uncertainty was measured, and a successful native-gate control does not establish that the ANN router preserves model quality.
