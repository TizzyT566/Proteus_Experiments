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

## Per-layer HDBSCAN results

The following results use Euclidean prototype scoring; they show how aggregate recall varies across layers. Cluster persistence is the HDBSCAN density-stability summary.

| Layer | Prototypes | Density clusters/expert | Noise | Mean persistence | Experts covered | Top-8 supported | Recall | Exact set | Calibrated weight L1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 960 | 3.276 | 29.9% | 0.411 | 243/256 (94.9%) | 99.4% | 83.2% | 43.4% | 0.646 |
| 1 | 948 | 3.333 | 33.8% | 0.468 | 240/256 (93.8%) | 99.9% | 74.7% | 33.8% | 0.772 |
| 2 | 959 | 3.198 | 40.7% | 0.476 | 252/256 (98.4%) | 100.0% | 70.4% | 32.4% | 0.885 |
| 3 | 908 | 3.045 | 40.7% | 0.502 | 244/256 (95.3%) | 99.8% | 71.1% | 32.7% | 0.836 |
| 4 | 933 | 3.124 | 43.0% | 0.492 | 242/256 (94.5%) | 99.1% | 77.6% | 34.3% | 0.766 |
| 5 | 964 | 3.144 | 53.1% | 0.484 | 243/256 (94.9%) | 99.6% | 69.9% | 30.2% | 0.845 |
| 6 | 929 | 3.404 | 43.3% | 0.460 | 223/256 (87.1%) | 99.8% | 70.2% | 30.2% | 0.828 |
| 7 | 946 | 3.325 | 47.1% | 0.432 | 231/256 (90.2%) | 99.6% | 70.8% | 31.9% | 0.806 |
| 8 | 938 | 3.400 | 47.5% | 0.452 | 225/256 (87.9%) | 96.9% | 69.2% | 28.7% | 0.824 |
| 9 | 936 | 3.243 | 47.4% | 0.423 | 235/256 (91.8%) | 100.0% | 70.2% | 29.0% | 0.842 |
| 10 | 974 | 3.325 | 45.5% | 0.387 | 234/256 (91.4%) | 99.9% | 72.7% | 30.4% | 0.772 |
| 11 | 946 | 3.244 | 44.8% | 0.413 | 234/256 (91.4%) | 99.6% | 72.9% | 30.1% | 0.784 |
| 12 | 937 | 2.940 | 46.1% | 0.406 | 249/256 (97.3%) | 100.0% | 69.8% | 31.8% | 0.844 |
| 13 | 951 | 3.315 | 44.7% | 0.379 | 232/256 (90.6%) | 99.9% | 66.6% | 27.9% | 0.882 |
| 14 | 967 | 3.254 | 37.9% | 0.369 | 244/256 (95.3%) | 99.8% | 65.3% | 27.9% | 0.870 |
| 15 | 954 | 3.151 | 39.3% | 0.360 | 245/256 (95.7%) | 99.6% | 66.6% | 27.7% | 0.839 |
| 16 | 930 | 3.170 | 47.5% | 0.336 | 235/256 (91.8%) | 100.0% | 69.3% | 27.8% | 0.849 |
| 17 | 904 | 3.073 | 50.9% | 0.336 | 233/256 (91.0%) | 98.8% | 63.4% | 27.6% | 0.940 |
| 18 | 941 | 3.202 | 41.5% | 0.334 | 238/256 (93.0%) | 99.8% | 64.1% | 27.6% | 0.902 |
| 19 | 927 | 3.178 | 44.0% | 0.347 | 236/256 (92.2%) | 100.0% | 64.5% | 25.6% | 0.890 |
| 20 | 950 | 3.235 | 41.0% | 0.346 | 238/256 (93.0%) | 99.0% | 64.0% | 25.5% | 0.900 |
| 21 | 962 | 3.316 | 45.6% | 0.340 | 237/256 (92.6%) | 99.7% | 65.9% | 26.8% | 0.961 |
| 22 | 938 | 3.146 | 45.3% | 0.384 | 240/256 (93.8%) | 100.0% | 71.2% | 29.0% | 0.775 |
| 23 | 936 | 3.024 | 44.0% | 0.387 | 247/256 (96.5%) | 100.0% | 70.8% | 28.3% | 0.803 |
| 24 | 933 | 2.948 | 54.0% | 0.383 | 248/256 (96.9%) | 99.8% | 67.8% | 27.1% | 0.896 |
| 25 | 925 | 3.079 | 45.5% | 0.401 | 240/256 (93.8%) | 100.0% | 66.9% | 25.1% | 0.903 |
| 26 | 940 | 3.004 | 48.6% | 0.407 | 249/256 (97.3%) | 100.0% | 65.6% | 27.1% | 0.878 |
| 27 | 935 | 2.964 | 49.0% | 0.392 | 250/256 (97.7%) | 99.8% | 65.8% | 25.6% | 0.855 |
| 28 | 980 | 3.468 | 47.5% | 0.369 | 231/256 (90.2%) | 99.9% | 69.3% | 27.4% | 0.893 |
| 29 | 956 | 3.296 | 52.2% | 0.403 | 233/256 (91.0%) | 99.6% | 68.4% | 28.0% | 0.826 |
| 30 | 941 | 3.295 | 43.0% | 0.431 | 234/256 (91.4%) | 100.0% | 69.3% | 26.6% | 0.806 |
| 31 | 962 | 3.402 | 48.8% | 0.409 | 229/256 (89.5%) | 100.0% | 69.0% | 27.4% | 0.792 |
| 32 | 951 | 3.493 | 45.2% | 0.454 | 221/256 (86.3%) | 99.8% | 75.0% | 28.5% | 0.706 |
| 33 | 944 | 3.336 | 43.6% | 0.466 | 229/256 (89.5%) | 99.5% | 75.4% | 30.4% | 0.743 |
| 34 | 934 | 3.594 | 38.1% | 0.463 | 212/256 (82.8%) | 99.6% | 78.6% | 32.1% | 0.631 |
| 35 | 926 | 3.279 | 44.0% | 0.455 | 229/256 (89.5%) | 100.0% | 68.5% | 26.1% | 0.788 |
| 36 | 937 | 3.437 | 45.3% | 0.445 | 222/256 (86.7%) | 99.7% | 70.5% | 26.5% | 0.745 |
| 37 | 951 | 3.452 | 47.0% | 0.444 | 221/256 (86.3%) | 99.6% | 72.8% | 26.4% | 0.683 |
| 38 | 946 | 3.740 | 44.8% | 0.453 | 208/256 (81.2%) | 99.4% | 73.1% | 29.0% | 0.757 |
| 39 | 982 | 3.806 | 50.3% | 0.455 | 211/256 (82.4%) | 99.9% | 77.5% | 32.4% | 0.605 |

## Evaluation details and metric definitions

The corpus yields 8,592 training and 2,172 held-out router-input states per layer for this split. HDBSCAN clustering is performed separately for each layer/expert, using only training states. Its data-driven density clusters plus stated residual/fallback handling define the prototypes; min_cluster_size and min_samples are density controls, not direct routing-accuracy objectives.

Top-8 recall is overlap with the native selected set divided by eight. Exact set is the fraction of held-out states with identical expert sets. Calibrated weight L1 is the absolute error between dense predicted and native gate distributions after temperature calibration on training states; zero is perfect and two is the maximum for normalized distributions. Support records whether all native top-8 experts for a held-out state were represented by training prototypes. Recall and weight L1 include unsupported held-out states, while support diagnoses how often this condition occurs.

The reported noise fraction is measured before fallback assignment. HDBSCAN discovered 3.05–3.81 density clusters per represented expert across the 40 layers, but that mean hides the wide cluster-count range across individual experts.

## Interpretation and limits

This full-depth run supports variable-count clustering as a promising offline routing proxy on this split. It does not establish stable prototype counts across datasets or checkpoints. Only 10 independent topics were held out, and residual noise prototypes can be broad representations.
