# Full-depth HDBSCAN prototypes in PCA-64

**Recorded completion time:** 2026-10-07 17:15:43 EDT. This filename time is based on the final result summary’s local modification time; an exact run start time was not recorded.

## Objective and method

This experiment extended variable-count HDBSCAN prototypes to all 40 layers and 256 experts per layer. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, a post-trained proxy checkpoint, and the corpus of 192 variants from 48 source topics (38 fit, 10 held out).

HDBSCAN was fitted separately per layer and expert in a train-only 64-D PCA space, while prototypes were means in the original 2,048-D hidden space. Settings were min_cluster_size 32, min_samples 5, and EOM. Large noise sets received a residual prototype; small noise sets were assigned to a discovered cluster; no-cluster cases used a global fallback. Metrics are macro-averages across 40 layers on one topic-group split.

## Results

| Scoring | Mean prototypes/layer | Top-8 recall | Exact set | Calibrated weight L1 |
| --- | ---: | ---: | ---: | ---: |
| Cosine | 944.5 | 70.2% | 29.0% | 0.814 |
| Dot product | 944.5 | 59.1% | 26.4% | 1.005 |
| Euclidean | 944.5 | 70.2% | 29.2% | 0.814 |
| Fixed Euclidean k-means + Euclidean scoring | 605.5 | 64.6% | 10.5% | 0.918 |

Euclidean HDBSCAN scoring improved recall on all 40 layers, by 5.6 percentage points on average over the fixed Euclidean baseline, while using about 339 more prototypes per layer. Noise averaged 44.8%. HDBSCAN chooses clusters by density stability; it does not directly optimize router accuracy.

## Interpretation and limits

This full-depth run supports variable-count clustering as a promising offline routing proxy on this split. It does not establish stable prototype counts across datasets or checkpoints. Only 10 independent topics were held out, and residual noise prototypes can be broad representations.

