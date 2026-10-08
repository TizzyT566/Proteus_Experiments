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

## Per-layer HDBSCAN results

The table reports full-dimensional Euclidean scoring. It complements the layer-macro averages above and shows the uneven layer-level response.

| Layer | Prototypes | Density clusters/expert | Noise | Mean persistence | Experts covered | Top-8 supported | Recall | Exact set | Calibrated weight L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 947 | 3.004 | 48.0% | 0.535 | 243/256 (94.9%) | 99.4% | 82.7% | 41.8% | 0.655 |
| 1 | 923 | 2.962 | 50.4% | 0.551 | 240/256 (93.8%) | 99.9% | 73.4% | 33.1% | 0.794 |
| 2 | 909 | 2.730 | 59.7% | 0.553 | 252/256 (98.4%) | 100.0% | 69.6% | 30.7% | 0.900 |
| 3 | 891 | 2.742 | 61.0% | 0.570 | 244/256 (95.3%) | 99.8% | 70.6% | 31.4% | 0.846 |
| 4 | 897 | 2.764 | 61.7% | 0.544 | 242/256 (94.5%) | 99.1% | 76.8% | 34.0% | 0.780 |
| 5 | 897 | 2.733 | 69.8% | 0.552 | 243/256 (94.9%) | 99.6% | 70.5% | 29.8% | 0.832 |
| 6 | 864 | 2.951 | 64.8% | 0.519 | 223/256 (87.1%) | 99.8% | 69.9% | 30.9% | 0.833 |
| 7 | 870 | 2.853 | 66.1% | 0.479 | 231/256 (90.2%) | 99.6% | 70.1% | 31.9% | 0.817 |
| 8 | 874 | 2.982 | 65.4% | 0.464 | 225/256 (87.9%) | 96.9% | 67.7% | 27.4% | 0.852 |
| 9 | 889 | 2.889 | 61.8% | 0.413 | 235/256 (91.8%) | 100.0% | 69.5% | 29.0% | 0.853 |
| 10 | 890 | 2.885 | 65.5% | 0.393 | 234/256 (91.4%) | 99.9% | 72.7% | 29.3% | 0.767 |
| 11 | 876 | 2.812 | 65.6% | 0.428 | 234/256 (91.4%) | 99.6% | 73.3% | 29.0% | 0.775 |
| 12 | 886 | 2.622 | 66.9% | 0.402 | 249/256 (97.3%) | 100.0% | 72.0% | 31.3% | 0.810 |
| 13 | 901 | 2.948 | 59.9% | 0.394 | 232/256 (90.6%) | 99.9% | 68.3% | 27.8% | 0.855 |
| 14 | 937 | 2.975 | 54.6% | 0.369 | 244/256 (95.3%) | 99.8% | 67.7% | 27.3% | 0.828 |
| 15 | 936 | 2.951 | 53.8% | 0.346 | 245/256 (95.7%) | 99.6% | 68.6% | 27.1% | 0.804 |
| 16 | 891 | 2.872 | 62.9% | 0.335 | 235/256 (91.8%) | 100.0% | 70.3% | 27.5% | 0.835 |
| 17 | 919 | 3.034 | 63.5% | 0.320 | 233/256 (91.0%) | 98.8% | 67.8% | 27.4% | 0.851 |
| 18 | 914 | 2.966 | 56.8% | 0.320 | 238/256 (93.0%) | 99.8% | 67.7% | 26.6% | 0.832 |
| 19 | 929 | 3.034 | 55.5% | 0.329 | 236/256 (92.2%) | 100.0% | 67.0% | 25.5% | 0.843 |
| 20 | 903 | 2.899 | 58.1% | 0.360 | 238/256 (93.0%) | 99.0% | 65.4% | 25.6% | 0.876 |
| 21 | 925 | 3.017 | 58.1% | 0.339 | 237/256 (92.6%) | 99.7% | 67.6% | 26.7% | 0.930 |
| 22 | 876 | 2.779 | 59.0% | 0.398 | 240/256 (93.8%) | 100.0% | 71.4% | 29.0% | 0.772 |
| 23 | 904 | 2.741 | 62.2% | 0.380 | 247/256 (96.5%) | 100.0% | 71.0% | 27.7% | 0.796 |
| 24 | 894 | 2.690 | 68.4% | 0.370 | 248/256 (96.9%) | 99.8% | 68.0% | 25.9% | 0.897 |
| 25 | 898 | 2.833 | 61.4% | 0.386 | 240/256 (93.8%) | 100.0% | 67.2% | 25.9% | 0.902 |
| 26 | 907 | 2.735 | 64.6% | 0.404 | 249/256 (97.3%) | 100.0% | 67.2% | 26.9% | 0.850 |
| 27 | 911 | 2.708 | 65.9% | 0.402 | 250/256 (97.7%) | 99.8% | 67.2% | 25.3% | 0.829 |
| 28 | 883 | 2.896 | 63.1% | 0.405 | 231/256 (90.2%) | 99.9% | 68.5% | 27.9% | 0.916 |
| 29 | 891 | 2.906 | 65.8% | 0.399 | 233/256 (91.0%) | 99.6% | 68.3% | 27.4% | 0.828 |
| 30 | 910 | 2.970 | 58.9% | 0.432 | 234/256 (91.4%) | 100.0% | 69.9% | 27.4% | 0.798 |
| 31 | 908 | 3.013 | 63.5% | 0.434 | 229/256 (89.5%) | 100.0% | 67.5% | 26.6% | 0.820 |
| 32 | 873 | 3.005 | 61.6% | 0.468 | 221/256 (86.3%) | 99.8% | 74.3% | 28.3% | 0.712 |
| 33 | 897 | 3.000 | 58.4% | 0.455 | 229/256 (89.5%) | 99.5% | 74.7% | 31.3% | 0.758 |
| 34 | 892 | 3.278 | 54.8% | 0.464 | 212/256 (82.8%) | 99.6% | 78.7% | 31.0% | 0.628 |
| 35 | 871 | 2.860 | 64.7% | 0.473 | 229/256 (89.5%) | 100.0% | 69.0% | 27.2% | 0.777 |
| 36 | 862 | 2.946 | 65.5% | 0.464 | 222/256 (86.7%) | 99.7% | 70.7% | 25.8% | 0.740 |
| 37 | 868 | 2.982 | 62.7% | 0.438 | 221/256 (86.3%) | 99.6% | 72.0% | 25.7% | 0.695 |
| 38 | 900 | 3.389 | 61.6% | 0.416 | 208/256 (81.2%) | 99.4% | 73.0% | 28.8% | 0.762 |
| 39 | 910 | 3.370 | 64.1% | 0.453 | 211/256 (82.4%) | 99.9% | 78.4% | 31.4% | 0.586 |

## Evaluation details and metric definitions

Each layer contributes 8,592 training and 2,172 held-out router-input states in this split. HDBSCAN operates per layer and native expert directly on 2,048-D training states. The reported prototype counts include residual and fallback representations for training states labeled noise or experts with no discovered cluster.

Top-8 recall measures predicted/native set overlap divided by eight. Exact set is the exact-set fraction. Calibrated weight L1 is the dense 256-way distribution error after training-only temperature fitting, with zero perfect and two maximum for normalized distributions. Top-8 support indicates whether every native top-8 expert for a held-out state had a fitted prototype. Main recall and L1 include all held-out states; support is a diagnostic.

The high full-dimensional noise fraction does not imply that 61.4% of held-out states are unusable; the reported route still assigns residual/fallback prototypes to the training noise. It does mean a large share of the training density structure was not assigned to a stable HDBSCAN cluster.

## Interpretation and limits

The full-dimensional result suggests that eliminating PCA did not materially change aggregate recall in this single split. HDBSCAN’s high-dimensional noise fraction was substantial, and residual handling remained part of the method. Only 10 independent topics were held out; prototype-count stability and generated-quality effects were not evaluated.
