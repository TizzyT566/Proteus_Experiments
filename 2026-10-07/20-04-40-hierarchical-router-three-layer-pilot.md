# Three-layer hierarchical router-output prototype pilot

**Recorded completion time:** 2026-10-07 20:04:40 EDT. This filename time is based on the aggregate result timestamp; an exact run start time was not recorded.

## Objective and method

This pilot tested whether hidden-state regions could be subdivided using the native router’s output, so a prototype could carry both a state centroid and contribution weights. It covered layers 0, 20, and 39 on topic-group seed 23.

For each layer, train-only PCA-384 hidden states were grouped into parent regions with HDBSCAN. This subset used 192 prompt variants from 48 topics, with 38 topic groups for fitting and 10 held out. Each parent was then subdivided separately using either standardized raw 256-way router scores or dense native top-8 weight vectors. A child prototype stored the mean hidden state in PCA space and mean dense native gate weights. Held-out states used the nearest centroid. Both HDBSCAN stages used min_cluster_size 32, min_samples 5, and EOM. Parent noise remained a parent region; sufficiently large child noise groups were retained as a child group, so every training example was represented but fallback prototypes could span broad regions.

## Aggregate results across the three layers

Matched-K recall compares predicted top K with the K highest-weight native experts. Top-8 coverage keeps the prediction pool fixed at eight. Results are one split, so there is no across-split standard deviation.

| Child clustering | K | Matched-K recall | Teacher top-K covered in predicted top 8 | Exact set | Weight L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw scores | 1 | 48.30% | 68.55% | 48.30% | 1.0341 |
| Raw scores | 2 | 51.11% | 67.59% | 39.53% | 0.9941 |
| Raw scores | 3 | 54.04% | 66.51% | 36.49% | 0.9444 |
| Raw scores | 4 | 56.43% | 65.57% | 35.17% | 0.9023 |
| Raw scores | 5 | 57.21% | 64.36% | 34.02% | 0.8845 |
| Raw scores | 6 | 58.15% | 62.99% | 32.90% | 0.8684 |
| Raw scores | 7 | 58.62% | 61.52% | 30.52% | 0.8600 |
| Raw scores | 8 | 59.80% | 59.80% | 32.21% | 0.8460 |
| Native weights | 1 | 50.21% | 72.59% | 50.21% | 0.9957 |
| Native weights | 2 | 53.26% | 71.58% | 40.39% | 0.9514 |
| Native weights | 3 | 56.73% | 70.09% | 37.22% | 0.8916 |
| Native weights | 4 | 59.38% | 68.78% | 36.69% | 0.8456 |
| Native weights | 5 | 59.87% | 67.30% | 34.39% | 0.8323 |
| Native weights | 6 | 60.54% | 65.79% | 32.98% | 0.8196 |
| Native weights | 7 | 60.97% | 64.16% | 30.86% | 0.8118 |
| Native weights | 8 | 62.23% | 62.23% | 32.44% | 0.7972 |

The parent stage averaged 96 clusters per layer with 44.6% noise. Raw-score children produced 103 prototypes per layer and 56.6% child noise; native-weight children produced 116.3 prototypes per layer and 64.9% child noise.

## Layer-level checks at selected K

These rows show selected operating points for each sampled layer. The K=1 and K=8 rows reveal whether the aggregate result is consistent across depth; K=4 provides an intermediate point. L1 is the renormalized predicted/reference top-K weight distance used by this analysis.

| Layer | K | Child clustering | Recall | Top-8 coverage | Exact set | Weight L1 | Prototypes/layer | Parent noise | Child noise |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | Raw scores | 54.5% | 76.5% | 54.5% | 0.911 | 100 | 40.0% | 60.4% |
| 0 | 1 | Native weights | 56.4% | 84.4% | 56.4% | 0.872 | 115 | 40.0% | 66.4% |
| 0 | 4 | Raw scores | 63.5% | 72.8% | 42.4% | 0.767 | 100 | 40.0% | 60.4% |
| 0 | 4 | Native weights | 69.3% | 79.1% | 46.2% | 0.661 | 115 | 40.0% | 66.4% |
| 0 | 8 | Raw scores | 67.5% | 67.5% | 39.4% | 0.701 | 100 | 40.0% | 60.4% |
| 0 | 8 | Native weights | 72.4% | 72.4% | 39.7% | 0.607 | 115 | 40.0% | 66.4% |
| 20 | 1 | Raw scores | 40.7% | 57.8% | 40.7% | 1.187 | 97 | 52.2% | 55.0% |
| 20 | 1 | Native weights | 44.7% | 61.7% | 44.7% | 1.107 | 120 | 52.2% | 73.0% |
| 20 | 4 | Raw scores | 48.0% | 55.7% | 29.1% | 1.070 | 97 | 52.2% | 55.0% |
| 20 | 4 | Native weights | 51.0% | 59.1% | 29.6% | 1.007 | 120 | 52.2% | 73.0% |
| 20 | 8 | Raw scores | 51.1% | 51.1% | 27.3% | 1.024 | 97 | 52.2% | 55.0% |
| 20 | 8 | Native weights | 53.6% | 53.6% | 27.2% | 0.970 | 120 | 52.2% | 73.0% |
| 39 | 1 | Raw scores | 49.8% | 71.4% | 49.8% | 1.005 | 112 | 41.7% | 54.5% |
| 39 | 1 | Native weights | 49.6% | 71.6% | 49.6% | 1.008 | 114 | 41.7% | 55.4% |
| 39 | 4 | Raw scores | 57.8% | 68.3% | 34.0% | 0.870 | 112 | 41.7% | 54.5% |
| 39 | 4 | Native weights | 57.9% | 68.1% | 34.3% | 0.869 | 114 | 41.7% | 55.4% |
| 39 | 8 | Raw scores | 60.8% | 60.8% | 30.0% | 0.812 | 112 | 41.7% | 54.5% |
| 39 | 8 | Native weights | 60.7% | 60.7% | 30.4% | 0.814 | 114 | 41.7% | 55.4% |

The layer-0 result drives much of the native-weight variant’s gain. At layer 20, both child-clustering variants remain noticeably weaker and native-weight child noise reaches 73.0%. At layer 39, the raw-score and native-weight methods are nearly tied.

## Metric definitions and diagnostics

Matched-K recall is the overlap between the predicted top K and the K native experts with the largest gate weights, divided by K. Top-8 coverage holds the prediction pool at eight and measures whether those K teacher contributors appear in it. Exact set is the proportion of states where the two K-sized sets match.

Weight L1 compares the normalized K-weight distributions over the expert IDs, with zero perfect and two maximum. Native mass captured, in the aggregate table, measures native top-8 weight assigned to experts in the predicted set. Parent and child noise are measured before the stated noise-group retention rules. Final prototype counts include child groups that preserve training-state coverage.

The three layers provide different views: layer 0 has much stronger recall and lower parent noise than layer 20; layer 39 shows little difference between raw-score and native-weight children. The all-layer average should therefore not be interpreted as a uniform per-layer effect.

## Interpretation and limits

Native-weight child clustering outperformed raw-score child clustering on the sampled layers and improved top-8 recall over the earlier fixed-centroid pilot. The pilot is only three layers and one split; it does not establish the full-depth tradeoff. Exact-set rates decline from K=1 as more experts are included, even while matched-K recall rises.
