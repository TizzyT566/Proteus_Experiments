# Proteus Expert-Routing Experiments — 2026-10-07

## Experimental objective

These experiments evaluate whether per-layer hidden-state prototypes can reproduce a frozen MoE model's native expert selections and contribution weights. They compare prototype clustering, similarity scoring, PCA dimensionality, and hierarchical clustering of hidden states and router outputs.

## Executive summary

The experiments test whether layer-wise hidden-state prototypes can approximate a frozen MoE model's native expert routing. Results show that prototype retrieval is substantially better than random selection and recovers useful expert candidates, but it does not yet reproduce the base router closely enough to establish the Proteus routing method.

The main findings are:

- HDBSCAN with Euclidean prototype scoring reached about **70% top-8 recall** across all 40 layers, compared with about **65%** for the tested fixed-count Euclidean k-means baseline.
- Clustering in PCA-384 performed essentially as well as higher dimensions across three topic-group splits. **384-D is a reasonable experimental default**, not a proven universal optimum.
- The existing per-expert prototype method has strong top-8 candidate coverage (**83.3% for the teacher's top-1 expert**) but weak exact ranking (**16.9% top-1 matched recall**).
- Hierarchical hidden-state then router-output clustering substantially improved top-1 ranking (**49.0%** using native weights), but reduced top-8 pool coverage and top-8 recall. It is a promising direction, not a replacement yet.
- The hierarchical run marked many examples as HDBSCAN noise. About **50%** were parent-cluster noise, with child noise averaging **59%** for raw router scores and **68%** for native weights.
- No experiment measured generated quality, end-to-end routing drift, expert hot-loading, VRAM/RAM/disk caching, or community-expert generalization.

## Model, traces, and evaluation protocol

- **Checkpoint actually used:** `Qwen/Qwen3.6-35B-A3B`, pinned revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- The public checkpoint used was **post-trained**, while the intended target is a Base checkpoint. Treat these results as a routing-method proxy, not validation on the exact target checkpoint.
- The model has **40 MoE layers**, **256 native experts per layer**, and captured **2048-D router-input hidden states**. The reference router uses top-8 in these experiments.
- Capture used BF16 model weights with GPU/CPU/disk offload; the router traces include full router scores, selected expert IDs, normalized top-8 weights, hidden states, and prompt/group IDs.
- The corpus contains **192 prompt variants from 48 source topics**. Variants within a topic are grouped to prevent topic leakage. Evaluations use 38 topic groups for fitting and 10 held out. The repeated evaluations use group split seeds 23, 42, and 71.
- Reported split standard deviations are across three group splits, not token-level confidence intervals. The three splits still share the same 48-topic corpus.
- Adaptive K was disabled; all routing comparisons are top-8 or report K=1…8 diagnostics.

## Experiment progression

### Initial three-layer pilot

Layers 0, 20, and 39 were captured over 48 prompt topics, with 38 topics for fitting and 10 held out. A fixed cosine k-means prototype baseline recovered **75.3%**, **49.0%**, and **70.0%** of the native top-8 experts at those layers. Layer 20 was the weak point. Prototype coverage was high, but exact top-8 set agreement was low (15.3%, 2.8%, and 5.3%). This established feasibility while showing that centroid similarity did not reproduce the router consistently.

### Comprehensive 40-layer prototype comparison

On the 40-layer, single-split evaluation, HDBSCAN was compared with a fixed-count baseline. HDBSCAN used `min_cluster_size=32`, `min_samples=5`, EOM selection, and a residual-prototype heuristic for noise. The initial full comparison used 64-D PCA for clustering and retained prototype means in the original 2048-D hidden-state space.

| Method | Mean prototypes/layer | Top-8 recall | Exact top-8 set | Calibrated weight L1 |
|---|---:|---:|---:|---:|
| HDBSCAN + cosine score | 944.5 | 70.2% | 29.0% | 0.814 |
| HDBSCAN + Euclidean score | 944.5 | 70.2% | 29.2% | 0.814 |
| Fixed Euclidean k-means + Euclidean score | 605.5 | 64.6% | 10.5% | 0.918 |

HDBSCAN + Euclidean scoring improved recall over the fixed-count Euclidean baseline on all 40 layers in this split. The result used more prototypes per layer. Layer-level HDBSCAN top-8 recall ranged from about 63% to 83%.

### Full 2048-D HDBSCAN run

HDBSCAN was also run directly in the full hidden-state dimension. It reached **70.7% top-8 recall**, **28.8% exact top-8 set accuracy**, and **0.806 calibrated weight L1**, with about **898 prototypes/layer**. This was only an initial split, not a repeat-split confirmation. Its mean noise fraction was about **61.4%**, higher than the 64-D run's approximately **44.8%**.

### PCA dimensionality sweep

The broad seed-23 sweep compared PCA dimensions 16 through 2048. Top-8 recall stayed close to 70–71% across settings, while noise and prototype counts changed. Across three split seeds, the main repeated settings were:

| PCA dimensions | Top-8 recall, mean ± split SD | Exact set, mean ± split SD | Weight L1, mean ± split SD | Prototypes/layer |
|---:|---:|---:|---:|---:|
| 128 | 69.51% ± 1.17% | 27.68% ± 1.35% | 0.8242 ± 0.0181 | 958.9 |
| 256 | 69.75% ± 1.24% | 27.54% ± 1.28% | 0.8200 ± 0.0195 | 935.9 |
| **384** | **69.83% ± 1.22%** | **27.55% ± 1.28%** | **0.8187 ± 0.0191** | 919.4 |
| 512 | 69.83% ± 1.23% | 27.50% ± 1.30% | 0.8187 ± 0.0194 | 909.5 |

384-D and 512-D were indistinguishable within split variation; 256-D was close. The original per-expert pipeline used PCA for clustering but stored centroids in 2048-D. The later hierarchical simulation instead clustered and matched in the 384-D PCA space. These are different prototype representations and should not be conflated.

### Top-K behavior of the 384-D per-expert baseline

Matched-K recall is overlap divided by K. Fixed top-8 pool coverage asks whether the teacher's top-K contributors appear anywhere in the predicted top-8 candidate pool.

| K | Matched-K recall | Teacher top-K covered by predicted top 8 | Exact set |
|---:|---:|---:|---:|
| 1 | 16.90% ± 0.06% | 83.29% ± 1.38% | 16.90% ± 0.06% |
| 2 | 29.26% ± 0.37% | 82.31% ± 1.31% | 6.25% ± 0.14% |
| 3 | 39.40% ± 0.41% | 80.69% ± 1.29% | 3.33% ± 0.04% |
| 4 | 48.06% ± 0.62% | 78.79% ± 1.28% | 2.39% ± 0.10% |
| 5 | 55.24% ± 0.82% | 76.71% ± 1.29% | 2.35% ± 0.09% |
| 6 | 61.37% ± 1.00% | 74.59% ± 1.30% | 3.90% ± 0.08% |
| 7 | 66.36% ± 1.13% | 72.36% ± 1.26% | 10.58% ± 0.52% |
| 8 | 69.83% ± 1.22% | 69.83% ± 1.22% | 27.55% ± 1.28% |

Matched-K recall can rise as K increases because it is normalized by K; smaller-K recall is not mathematically guaranteed to be higher. Fixed-pool coverage is the metric that declines as lower-ranked teacher contributors are included.

### Hierarchical hidden-state / router-output prototypes

The full hierarchical test used train-only PCA-384 HDBSCAN parent clusters, then child HDBSCAN clusters within each parent. Child clustering used either standardized raw router score vectors or dense native top-8 gate-weight vectors. Each final child cluster was represented by its PCA-space hidden-state centroid and mean native weight vector. Held-out states were assigned to their single nearest centroid. HDBSCAN parameters were `min_cluster_size=32`, `min_samples=5`, EOM.

| K | Per-expert baseline recall | Raw-score child recall | Native-weight child recall | Baseline top-8 coverage | Raw-score top-8 coverage | Native-weight top-8 coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.9% ± 0.1% | 47.0% ± 1.7% | **49.0% ± 1.7%** | **83.3% ± 1.4%** | 67.8% ± 2.1% | 70.6% ± 1.9% |
| 2 | 29.3% ± 0.4% | 49.7% ± 1.6% | **52.1% ± 1.4%** | **82.3% ± 1.3%** | 65.9% ± 1.8% | 68.7% ± 1.6% |
| 3 | 39.4% ± 0.4% | 51.5% ± 1.5% | **54.0% ± 1.3%** | **80.7% ± 1.3%** | 64.1% ± 1.7% | 66.9% ± 1.4% |
| 4 | 48.1% ± 0.6% | 52.8% ± 1.5% | **55.3% ± 1.3%** | **78.8% ± 1.3%** | 62.5% ± 1.6% | 65.2% ± 1.4% |
| 5 | **55.2% ± 0.8%** | 54.0% ± 1.5% | **56.4% ± 1.3%** | **76.7% ± 1.3%** | 60.9% ± 1.6% | 63.5% ± 1.3% |
| 6 | **61.4% ± 1.0%** | 54.7% ± 1.5% | 57.1% ± 1.3% | **74.6% ± 1.3%** | 59.3% ± 1.6% | 61.8% ± 1.3% |
| 7 | **66.4% ± 1.1%** | 55.2% ± 1.4% | 57.5% ± 1.2% | **72.4% ± 1.3%** | 57.7% ± 1.5% | 60.1% ± 1.2% |
| 8 | **69.8% ± 1.2%** | 55.9% ± 1.5% | 58.1% ± 1.2% | **69.8% ± 1.2%** | 55.9% ± 1.5% | 58.1% ± 1.2% |

The native-weight variant consistently edged the raw-score variant. At K=1 it improved exact selection from 16.9% to 49.0%, but the predicted top-8 pool covered only 70.6% of teacher top-1 contributors, below the baseline's 83.3%. At K=8, matched recall was lower than baseline (58.1% versus 69.8%), while exact-set accuracy was similar (28.4% versus 27.5%).

The hierarchical experiment averaged 91 parent clusters/layer and 97.3 raw-score or 104.7 native-weight prototypes/layer. Parent noise was 50.0%; child noise averaged 59.4% and 67.9%, respectively. Noise was retained in fallback groups, which guarantees coverage but can make broad, less-specific prototypes.

## Interpretation and current limits

1. **Prototype retrieval is feasible but incomplete.** It finds useful candidate experts, but exact native route reconstruction remains difficult.
2. **PCA-384 is a practical candidate for offline clustering.** The measured gain over 256-D was small; 512-D did not improve quality. This does not establish the correct dimension for the eventual shared usefulness embedding.
3. **The hierarchical split improves top-contributor ranking but hurts candidate-pool coverage.** Hard assignment to one final centroid is a likely limitation to examine next.
4. **Native selected IDs and normalized gate weights are the most direct reference for actual routing.** Raw score vectors are useful diagnostics but should not be assumed to equal the executed top-8 decision.
5. **The router prototype experiments are not yet a test of the community expert architecture.** No independently contributed/finetuned expert was installed or evaluated. No common frozen usefulness encoder was tested.
6. **Generalization is uncertain.** The corpus has only 48 independent topic groups, the checkpoint is post-trained rather than the named Base checkpoint, and the experiments are offline.
7. **Runtime design remains untested.** Dynamic registry updates, hot loading, VRAM residency limits, RAM/disk tiers, and cache eviction were not measured. Adaptive K was intentionally deferred.

## Self-contained method notes

### Per-expert HDBSCAN prototype procedure

For each layer and topic-group split:

1. Hold out complete topic groups; fit dimensionality reduction and all calibration only on training groups.
2. Project the 2048-D router-input hidden states to the tested PCA dimension.
3. For each reference expert, cluster its training hidden states with HDBSCAN using Euclidean distance, `min_cluster_size=32`, `min_samples=5`, and EOM cluster selection.
4. Represent each discovered cluster by the mean of its member hidden states in the original 2048-D space.
5. Handle noise with residual prototypes: small noise groups are assigned to a nearby discovered cluster, large noise groups receive a residual centroid, and an expert with no density cluster receives a fallback centroid.
6. Score a held-out hidden state against each expert's prototype set using the tested similarity/distance rule; max-pool prototype scores per expert, rank experts, and fit only the routing-weight temperature on the training split.

### Hierarchical hidden-state/router-output procedure

The following pseudocode describes the full PCA-384 experiment. It is illustrative and self-contained, not a runnable script.

```text
For each layer and topic-group split:
    Fit PCA(384) on training hidden states only.
    Transform train and held-out hidden states with that PCA.
    Cluster training hidden states with HDBSCAN to form parent regions.

    For each parent region, including the parent's noise points as one region:
        Choose child features:
            A. standardized 256-D raw router scores, or
            B. dense 256-D vectors of native top-8 expert weights.
        Cluster the child features with HDBSCAN.
        For each child group:
            prototype = mean(PCA-projected hidden state)
            route_vector = mean(dense native top-8 weight vector)

    For each held-out hidden state:
        Select its single nearest prototype by squared Euclidean distance.
        Use that prototype's route_vector to rank experts and compute K=1..8 metrics.
```

The hierarchical run used `min_cluster_size=32`, `min_samples=5`, and EOM selection for both clustering stages. Child-noise groups smaller than the minimum size were assigned to their nearest child center; larger noise sets were retained as residual child groups; when no child density clusters existed, the region became one fallback group. The hierarchical simulation's prototypes and distances are in PCA-384 space, unlike the original per-expert HDBSCAN pipeline, whose means were retained in 2048-D hidden space.

### Metric definitions

For each token, sort the reference's captured native top-8 experts by their normalized native gate weights. Let `T_K` be the first K reference experts and `P_K` the K highest-scoring predicted experts.

- **Matched-K recall:** `|T_K ∩ P_K| / K`, averaged over tokens and layers.
- **Top-8 pool coverage:** `|T_K ∩ P_8| / K`; the predicted candidate pool stays at 8 while the reference set varies from K=1 to K=8.
- **Exact-set rate:** the fraction of tokens for which `T_K` and `P_K` are identical.
- **Native weight mass captured:** sum of the reference's native top-8 weights assigned to experts in the predicted active set.
- **Weight L1:** sum of absolute differences between dense 256-expert predicted and reference gate distributions. The original per-expert evaluations fit a softmax temperature on training data; the hierarchical evaluation renormalizes the predicted and reference top-K weights for each K. Zero is a perfect match; two is the maximum.
- **Noise fraction:** fraction of training points labeled noise by HDBSCAN before the residual-group handling described above.

All averages are macro-averages across layers. Repeated-run `±` values are standard deviations across the three topic-group split seeds, not token-level confidence intervals.
