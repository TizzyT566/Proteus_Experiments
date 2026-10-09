# Affine expert prediction vectors from captured routing data

**Completed:** 2026-10-09 07:45 Eastern Daylight Time  
**Checkpoint:** `Qwen/Qwen3.6-35B-A3B` at `995ad96eacd98c81ed38be0c5b274b04031597b0`  
**Environment:** Windows, project `I:\Proteus`, PyTorch CUDA on the RTX 5090 (`cuda:0`).

## Result

This experiment replaces each fixed activation mean with a learned, input-dependent affine predictor for each layer/expert. It fits one 2,048D vector and one scalar bias per expert to predict that expert's native mixture contribution from the captured hidden state. There is no HDBSCAN clustering and no PCA.

Across 180,160 held-out layer-token positions, exhaustive predictor scoring reached **0.7878 top-8 recall**, **0.3482 exact-set rate**, and **0.4670 routing-weight L1**. This improves substantially over the previous contribution-weighted mean (**0.5960 / 0.0192 / 0.9740**) and the previous HDBSCAN/PCA-384 ANN router (**0.6938 / 0.2404 / 0.8272**).

The 128-candidate HNSW route reached **0.7770 recall**, **0.3081 exact-set rate**, and **0.4837 weight L1**. It included all eight native experts on **77.67%** of held-out positions. The exact predictor's own top 128 candidates cover all eight native experts on **90.71%** of positions, so HNSW retrieval is a second, measurable source of loss.

The INT4 generation checks were still poor. With the exact 256-expert route, the prompt stopped after 2 tokens (`<think>`). With HNSW-128, default stopping returned 2 tokens (`question`); forcing a 16-token minimum returned 17 tokens and stopped mid-sentence. Better routing metrics did not yield a useful answer on this check.

The custom lookup is also slower than the native gate at this 256-expert scale: about **109.8 ms** per token for 40 layers at HNSW-128 and **112.8 ms** for exact-256 search, versus **22.9 ms** and **21.7 ms** for the native GPU gate in the paired measurements. The current CPU index path is roughly **4.8–5.2× slower**.

## What was fit

For layer `l`, expert `e`, and captured hidden state `h`, fit:

```text
score(l,e,h) = v(l,e) · h + b(l,e)
```

The training target is the expert's saved native top-eight mixture weight when selected, and zero when not selected. Thus each row of training data supervises all 256 experts: eight nonzero targets and 248 zero targets. Ridge regression fits each output from the same captured inputs. An inner split of the outer training topic groups chooses the penalty from `1e-4, 1e-3, 1e-2, 1e-1, 1, 10` by top-eight recall, with exact-set rate as a tie-breaker. The selected penalty was `0.1` in 38 layers and `1.0` in two. The 10 held-out topic groups are not used for fitting or penalty selection.

At inference, scores rank experts; the selected top eight are softmax-normalized using a per-layer temperature calibrated on training data. The stored expert vector stays 2,048D; the ANN key appends its bias as one extra coordinate, producing a 2,049D inner-product key. The query appends a constant `1`, so the dot product equals `v · h + b`.

When all 256 candidates are requested, the runtime asks USearch for an exact inner-product search. Smaller candidate counts use HNSW. This distinction matters: the exact-256 result is the learned predictor's quality without ANN omissions; HNSW-128 measures the scalable shortlist path.

## Corpus and split

The experiment reuses the transferred corpus and saved profile traces: 384 prompts organized into 48 topic groups, with captured inputs and native routing output for all 40 layers. The outer split holds out whole topic groups (seed 23): 38 groups / 17,792 positions for fitting and calibration, and 10 groups / 4,504 positions for validation at each layer. This yields 180,160 held-out layer-token observations. Ridge selection uses a second topic-group split inside the outer training groups (30 inner-fit groups and 8 inner-validation groups).

The model created 10,240 layer/expert vectors, each 2,048D, plus one bias per vector. The compressed bank is 72.9 MB and the 40 HNSW files total 44.8 MB. The fit and index build took **293.0 seconds**; the full exact/HNSW held-out evaluation took **198.6 seconds**.

## Aggregate held-out comparison

| Router/scoring method | Top-8 recall | Exact set | Weight L1 | Native top-8 candidate support |
| --- | ---: | ---: | ---: | ---: |
| Affine predictors, exact score over all 256 | **0.7878** | **0.3482** | **0.4670** | 1.0000 |
| Affine predictors, HNSW-128 shortlist | 0.7770 | 0.3081 | 0.4837 | 0.7767 |
| Contribution-weighted 2,048D means, HNSW-128 | 0.5960 | 0.0192 | 0.9740 | 0.8989 |
| HDBSCAN/PCA-384, prior ANN router | 0.6938 | 0.2404 | 0.8272 | 0.8307 |

All methods use the same corpus and outer topic split. The affine predictor improves exact scores and HNSW-128 over the single-mean baseline. Its exhaustive top-8 candidate-support curve is:

| Exact predictor candidates retained | 8 | 16 | 32 | 64 | 128 | 192 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Positions containing all 8 native experts | 0.3482 | 0.5479 | 0.6811 | 0.8057 | 0.9071 | 0.9502 | 1.0000 |

This curve uses exact vector scores to isolate candidate-count loss. The separate HNSW-128 path covers all native top-eight experts on 77.67% of positions, lower than the 90.71% exact-score coverage at the same candidate count.

## Per-layer results

Layer IDs are zero-based. The first three columns use exhaustive affine scores over all 256 experts. HNSW recall uses 128 approximate candidates; support is the fraction of held-out positions for which all eight native experts appear among those candidates.

| Layer | Exact recall | Exact set | Exact weight L1 | HNSW-128 recall | HNSW-128 support |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8530 | 0.4714 | 0.3708 | 0.8524 | 0.9225 |
| 1 | 0.8186 | 0.4165 | 0.4259 | 0.8185 | 0.9349 |
| 2 | 0.7631 | 0.3803 | 0.5489 | 0.7622 | 0.8730 |
| 3 | 0.7529 | 0.3131 | 0.5686 | 0.7354 | 0.6698 |
| 4 | 0.8217 | 0.4076 | 0.4073 | 0.8191 | 0.8559 |
| 5 | 0.7760 | 0.3537 | 0.4890 | 0.7751 | 0.8353 |
| 6 | 0.7865 | 0.3732 | 0.4812 | 0.7848 | 0.8943 |
| 7 | 0.7721 | 0.3455 | 0.5147 | 0.7638 | 0.7813 |
| 8 | 0.7570 | 0.3401 | 0.5268 | 0.7562 | 0.8568 |
| 9 | 0.7792 | 0.3433 | 0.4874 | 0.7768 | 0.8748 |
| 10 | 0.8050 | 0.3615 | 0.4349 | 0.7991 | 0.8777 |
| 11 | 0.8120 | 0.3672 | 0.4270 | 0.7971 | 0.7436 |
| 12 | 0.7659 | 0.3563 | 0.5222 | 0.7641 | 0.8701 |
| 13 | 0.7718 | 0.3348 | 0.5004 | 0.7627 | 0.8126 |
| 14 | 0.7442 | 0.3384 | 0.5724 | 0.7411 | 0.8333 |
| 15 | 0.7619 | 0.3151 | 0.5352 | 0.7479 | 0.7642 |
| 16 | 0.7720 | 0.3421 | 0.5129 | 0.7698 | 0.8450 |
| 17 | 0.7666 | 0.3552 | 0.5195 | 0.7612 | 0.7942 |
| 18 | 0.7572 | 0.3217 | 0.5268 | 0.7493 | 0.7260 |
| 19 | 0.7302 | 0.3153 | 0.5749 | 0.7229 | 0.7622 |
| 20 | 0.7385 | 0.3068 | 0.5753 | 0.7328 | 0.7611 |
| 21 | 0.7683 | 0.3188 | 0.5219 | 0.7608 | 0.7687 |
| 22 | 0.8061 | 0.3208 | 0.4611 | 0.7764 | 0.6823 |
| 23 | 0.7865 | 0.3228 | 0.4573 | 0.7816 | 0.8155 |
| 24 | 0.7650 | 0.3282 | 0.5217 | 0.7508 | 0.6801 |
| 25 | 0.7868 | 0.3222 | 0.4496 | 0.7741 | 0.6610 |
| 26 | 0.7560 | 0.3315 | 0.5206 | 0.7516 | 0.7638 |
| 27 | 0.7606 | 0.2991 | 0.5213 | 0.7595 | 0.8601 |
| 28 | 0.7910 | 0.3262 | 0.4252 | 0.7806 | 0.7271 |
| 29 | 0.7835 | 0.3399 | 0.4571 | 0.7698 | 0.7493 |
| 30 | 0.7889 | 0.3142 | 0.4434 | 0.7809 | 0.8206 |
| 31 | 0.7868 | 0.3273 | 0.4472 | 0.7726 | 0.6785 |
| 32 | 0.8353 | 0.3561 | 0.3508 | 0.7992 | 0.5995 |
| 33 | 0.8393 | 0.3839 | 0.3520 | 0.8270 | 0.7933 |
| 34 | 0.8598 | 0.3994 | 0.3025 | 0.8434 | 0.7742 |
| 35 | 0.7749 | 0.3097 | 0.4688 | 0.7424 | 0.5695 |
| 36 | 0.8015 | 0.3313 | 0.4201 | 0.7727 | 0.6630 |
| 37 | 0.8151 | 0.3488 | 0.3976 | 0.7949 | 0.6630 |
| 38 | 0.8378 | 0.3710 | 0.3537 | 0.8243 | 0.7962 |
| 39 | 0.8627 | 0.4167 | 0.2844 | 0.8249 | 0.7134 |

## INT4 generation checks

All runs used the same prompt, greedy decode, and 48-token maximum:

```text
Explain how PCA can help a nearest-neighbor router using a short example.
```

| Router mode | Length control | New tokens | Decode time | Generated text |
| --- | --- | ---: | ---: | --- |
| Exact-256 affine prediction | Default | 2 | 16.82 s | `<think>` |
| HNSW-128 affine prediction | Default | 2 | 18.16 s | `question` |
| HNSW-128 affine prediction | Minimum 16 | 17 | 35.62 s | `question is asking for an explanation of how **PCA (nearest-neighbor router)` |

These outputs are not useful answers. They show that better per-layer routing agreement alone is insufficient to make the end-to-end generation reliable. Routing errors accumulate across 40 layers, and this model was trained to imitate gate mixture weights on prompt-profile states rather than optimize text loss under the replacement router.

## Router-only latency

Paired routing-only tests used 32 held-out states per layer, two repeats, CUDA synchronization after every layer decision, and no expert MLP or other model work.

| Route | Median per layer decision | Median over 40 layers per token | Top-8 sets matching native (640 samples) |
| --- | ---: | ---: | ---: |
| Native quantized GPU gate | 0.572 ms (128 run); 0.543 ms (256 run) | 22.87 ms; 21.74 ms | — |
| Affine prediction + HNSW-128 | 2.745 ms | 109.80 ms | 10.63% |
| Affine prediction + exact-256 | 2.819 ms | 112.78 ms | 11.56% |

Both custom routes were about five times slower than native at 256 experts. The HNSW lookup transfers the hidden vector to the CPU and runs a high-dimensional index query; at this small registry size, the native GPU gate is very efficient. This experiment does not show the crossover point where ANN might become faster for a much larger registry.

## What this says about the design

The fixed contribution-weighted mean underperformed because it assigned one activation-space address to each expert. The affine predictor learns a decision surface instead: a hidden state can raise or lower an expert's score based on its direction along 2,048 learned coordinates. That is why its routing statistics improve substantially.

This target is still a **proxy for the original gate's behavior**, not a counterfactual measure of expert usefulness. It learns the mixture weight the native router assigned after its own top-eight selection. It does not measure whether an expert would improve the model output or loss if substituted into a different registry.

The data also does not prove that independently trained packages from unrelated experts are directly comparable. A registry can store one vector and bias per layer/expert and append packages without refitting all rows, but each package needs compatible per-layer hidden-state semantics, a consistently trained target, and compatible score calibration. The current experiment used one known 256-expert model and did not test newly trained or unrelated expert packages.

Storage remains proportional to the number of experts: one 2,048D vector plus bias per layer/expert. ANN changes how candidates are searched; it does not remove the vectors. The HNSW-128 test shows the current shortlist implementation also misses native experts. A next experiment should improve large-registry retrieval and train/evaluate predictors against a held-out mixture-weight objective while checking generation quality at every layer.

## Reproduction

Build all layer predictors and the exact-256 inner-product indexes:

```powershell
& .\.venv\Scripts\python.exe experiments\build_affine_prediction_router.py `
  --trace-dir traces\expert_profile_qwen36 `
  --output-dir results\affine_prediction_router_qwen36_2048_ip_20261009 `
  --device cuda:0 --ann-neighbors 256
```

Evaluate exact scores plus an approximate HNSW-128 shortlist, then synchronize the deployed router report and temperatures:

```powershell
& .\.venv\Scripts\python.exe experiments\evaluate_affine_prediction_router.py `
  --router-dir results\affine_prediction_router_qwen36_2048_ip_20261009 `
  --ann-count 128 --device cuda:0
& .\.venv\Scripts\python.exe experiments\finalize_affine_prediction_router.py
```

Run the generation controls and router-only timing:

```powershell
& .\.venv\Scripts\python.exe experiments\infer_qwen_moe_int4_ann.py `
  --router-dir results\affine_prediction_router_qwen36_2048_ip_20261009 `
  --ann-neighbors 256 `
  --prompt "Explain how PCA can help a nearest-neighbor router using a short example." `
  --max-new-tokens 48

& .\.venv\Scripts\python.exe experiments\infer_qwen_moe_int4_ann.py `
  --router-dir results\affine_prediction_router_qwen36_2048_ip_20261009 `
  --ann-neighbors 128 --also-min-new-tokens 16 `
  --prompt "Explain how PCA can help a nearest-neighbor router using a short example." `
  --max-new-tokens 48

& .\.venv\Scripts\python.exe experiments\benchmark_weighted_vector_router.py `
  --router-dir results\affine_prediction_router_qwen36_2048_ip_20261009 `
  --ann-neighbors 128 --contexts 32 --repeats 2 `
  --output results\affine_prediction_router_only_128_20261009.json
```

## Artifacts

- Builder: [`build_affine_prediction_router.py`](../../experiments/build_affine_prediction_router.py)
- Runtime: [`ann_router_runtime.py`](../../experiments/ann_router_runtime.py)
- Held-out metrics, candidate-support curve, and per-layer details: [`validation_report.json`](../../results/affine_prediction_router_qwen36_2048_ip_20261009/validation_report.json)
- Router metadata: [`router_report.json`](../../results/affine_prediction_router_qwen36_2048_ip_20261009/router_report.json)
- Vector and bias bank: [`affine_prediction_expert_bank.npz`](../../results/affine_prediction_router_qwen36_2048_ip_20261009/affine_prediction_expert_bank.npz)
- Router-only timing: [`128 candidates`](../../results/affine_prediction_router_only_128_20261009.json), [`exact 256`](../../results/affine_prediction_router_only_exact256_20261009.json)
- Generation logs: [`exact 256`](../../artifacts/runlogs/affine_prediction_generation_exact256_20261009.out.log), [`HNSW-128 and min-16`](../../artifacts/runlogs/affine_prediction_generation_hnsw128_20261009.out.log)

This is one checkpoint revision and one outer topic split; no repeated-seed confidence interval was run. Captured states came from corpus prompts rather than free-running generations.
