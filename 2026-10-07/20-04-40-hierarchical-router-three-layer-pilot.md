# Three-layer hierarchical router-output prototype pilot

**Recorded completion time:** 2026-10-07 20:04:40 EDT. This filename time is based on the aggregate result timestamp; an exact run start time was not recorded.

## Objective and method

This pilot tested whether hidden-state regions could be subdivided using the native router’s output, so a prototype could carry both a state centroid and contribution weights. It covered layers 0, 20, and 39 on topic-group seed 23.

For each layer, train-only PCA-384 hidden states were grouped into parent regions with HDBSCAN. Each parent was then subdivided separately using either standardized raw 256-way router scores or dense native top-8 weight vectors. A child prototype stored the mean hidden state in PCA space and mean dense native gate weights. Held-out states used the nearest centroid. Both HDBSCAN stages used min_cluster_size 32, min_samples 5, and EOM.

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

## Interpretation and limits

Native-weight child clustering outperformed raw-score child clustering on the sampled layers and improved top-8 recall over the earlier fixed-centroid pilot. The pilot is only three layers and one split; it does not establish the full-depth tradeoff. Exact-set rates decline from K=1 as more experts are included, even while matched-K recall rises.

