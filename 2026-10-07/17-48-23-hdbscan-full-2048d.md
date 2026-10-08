# Full-dimensional HDBSCAN prototypes

**Recorded completion time:** 2026-10-07 17:48:23 EDT. This filename time is based on the final result summary’s local modification time; an exact run start time was not recorded.

## Objective and method

This run repeated the full-depth HDBSCAN prototype evaluation without PCA, clustering directly in the original 2,048-dimensional hidden-state space. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, and the same corpus and topic-group split as the 64-D HDBSCAN evaluation: 192 prompt variants from 48 source topics, with 38 topics for fitting and 10 held out.

HDBSCAN was fitted per layer and expert using min_cluster_size 32, min_samples 5, and EOM. Prototypes were means in original hidden-state space. Large noise sets received residual prototypes, small noise sets were assigned to discovered clusters, and no-cluster cases received a global fallback. The reported metrics are macro-averages over 40 layers on one split.

## Results

| Scoring | Mean prototypes/layer | Top-8 recall | Exact set | Calibrated weight L1 |
| --- | ---: | ---: | ---: | ---: |
| Cosine | 898.1 | 70.4% | 28.6% | 0.810 |
| Dot product | 898.1 | 58.8% | 24.5% | 1.009 |
| Euclidean | 898.1 | 70.7% | 28.8% | 0.806 |
| Fixed Euclidean k-means + Euclidean scoring | 605.5 | 64.6% | 10.5% | 0.918 |

Euclidean HDBSCAN scoring improved recall on all 40 layers by 6.1 percentage points on average over the fixed Euclidean baseline. Mean HDBSCAN noise was 61.4%, compared with 44.8% for the PCA-64 run. The direct full-dimensional result is close to the PCA-64 score on this split, but the two runs do not establish a dimension preference.

## Interpretation and limits

The full-dimensional result suggests that eliminating PCA did not materially change aggregate recall in this single split. HDBSCAN’s high-dimensional noise fraction was substantial, and residual handling remained part of the method. Only 10 independent topics were held out; prototype-count stability and generated-quality effects were not evaluated.

