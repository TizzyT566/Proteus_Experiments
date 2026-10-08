# Three-layer HDBSCAN prototype pilot

**Recorded completion time:** 2026-10-07 17:07:52 EDT. This filename time is based on the final result artifact’s local modification time; an exact run start time was not recorded.

## Objective and method

This follow-up tested variable-count density clustering on the same three representative layers as the initial pilot: 0, 20, and 39. For each layer and native expert, HDBSCAN discovered clusters in a train-only 64-dimensional PCA projection; prototype vectors were means in the original hidden-state space.

The run used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, with min_cluster_size 32, min_samples 5, and EOM selection. Large noise sets received one residual prototype; small noise sets were assigned to their nearest discovered cluster, and experts with no clusters received a global fallback. The same topic-group split as the fixed-centroid baseline was used, with calibration fitted on training tokens.

## Results

| Layer | Prototypes | Mean clusters/expert | Noise fraction | Top-8 supported | Cosine recall | Dot-product recall | Euclidean recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 960 | 3.28 | 29.9% | 99.4% | 83.1% | 72.4% | 83.2% |
| 20 | 950 | 3.24 | 41.0% | 99.0% | 64.6% | 53.6% | 64.0% |
| 39 | 982 | 3.81 | 50.3% | 99.9% | 77.1% | 66.7% | 77.5% |

## Interpretation and limits

The variable-count prototypes improved recall over the initial fixed-centroid pilot on all three sampled layers, particularly layer 20. This was a three-layer, single-split pilot and does not establish the full-depth result. Noise handling and the number of clusters depend on HDBSCAN settings; the cluster count was selected by density stability, not directly optimized for routing accuracy.

