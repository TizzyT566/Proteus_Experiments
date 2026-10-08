# Full-depth hierarchical hidden-state and router-output prototypes

**Recorded completion time:** 2026-10-07 20:32:03 EDT. This filename time is based on the final comparison summary’s local modification time; an exact run start time was not recorded.

## Objective and protocol

This experiment extended hierarchical clustering to all 40 layers and 256 experts per layer, comparing child clusters built from standardized raw router scores with child clusters built from dense native top-8 weights. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, and topic-group seeds 23, 42, and 71. Each split used 192 prompt variants from 48 source topics, with 38 groups for fitting and 10 held out, yielding 8,592 training and 2,172 held-out states per layer.

Within each layer, train-only PCA-384 hidden states were clustered into parent regions. Each parent region was subdivided using one of the two router-output representations. Raw router-score vectors were standardized before clustering; native-weight vectors were the dense 256-expert representation of native top-8 gate weights. A child prototype stored its mean hidden state in PCA space and the cluster’s mean native 256-expert weight vector. Held-out states selected the nearest centroid by squared Euclidean distance and used the stored weights to rank experts. Both HDBSCAN stages used min_cluster_size 32, min_samples 5, and EOM. Parent noise was retained as a region; sufficiently large child noise groups were retained as groups, ensuring training-state coverage but allowing broad fallback prototypes. Values are macro-averages over 40 layers; ± is standard deviation across three topic-group splits.

## Expert selection and candidate coverage

Matched-K recall compares the prediction’s top K against the teacher’s K highest-weight experts. Top-8 coverage uses a fixed eight-expert prediction pool.

| K | Baseline recall | Raw-score child recall | Native-weight child recall | Baseline top-8 coverage | Raw-score top-8 coverage | Native-weight top-8 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.9 ± 0.1% | 47.0 ± 1.7% | 49.0 ± 1.7% | 83.3 ± 1.4% | 67.8 ± 2.1% | 70.6 ± 1.9% |
| 2 | 29.3 ± 0.4% | 49.7 ± 1.6% | 52.1 ± 1.4% | 82.3 ± 1.3% | 65.9 ± 1.8% | 68.7 ± 1.6% |
| 3 | 39.4 ± 0.4% | 51.5 ± 1.5% | 54.0 ± 1.3% | 80.7 ± 1.3% | 64.1 ± 1.7% | 66.9 ± 1.4% |
| 4 | 48.1 ± 0.6% | 52.8 ± 1.5% | 55.3 ± 1.3% | 78.8 ± 1.3% | 62.5 ± 1.6% | 65.2 ± 1.4% |
| 5 | 55.2 ± 0.8% | 54.0 ± 1.5% | 56.4 ± 1.3% | 76.7 ± 1.3% | 60.9 ± 1.6% | 63.5 ± 1.3% |
| 6 | 61.4 ± 1.0% | 54.7 ± 1.5% | 57.1 ± 1.3% | 74.6 ± 1.3% | 59.3 ± 1.6% | 61.8 ± 1.3% |
| 7 | 66.4 ± 1.1% | 55.2 ± 1.4% | 57.5 ± 1.2% | 72.4 ± 1.3% | 57.7 ± 1.5% | 60.1 ± 1.2% |
| 8 | 69.8 ± 1.2% | 55.9 ± 1.5% | 58.1 ± 1.2% | 69.8 ± 1.2% | 55.9 ± 1.5% | 58.1 ± 1.2% |

Native-weight clustering had higher matched-K recall than raw-score clustering at every K, including 58.1% versus 55.9% at K=8. The per-expert baseline remained higher at K=8 (69.8%). Both hierarchical variants had lower top-8 candidate coverage than the baseline.

## Exact-set agreement

| K | Per-expert baseline | Raw-score child | Native-weight child |
| ---: | ---: | ---: | ---: |
| 1 | 16.9 ± 0.1% | 47.0 ± 1.7% | 49.0 ± 1.7% |
| 2 | 6.2 ± 0.1% | 35.8 ± 1.5% | 37.3 ± 1.4% |
| 3 | 3.3 ± 0.0% | 32.6 ± 1.4% | 33.4 ± 1.4% |
| 4 | 2.4 ± 0.1% | 30.7 ± 1.5% | 31.2 ± 1.5% |
| 5 | 2.4 ± 0.1% | 29.3 ± 1.5% | 29.6 ± 1.4% |
| 6 | 3.9 ± 0.1% | 28.3 ± 1.3% | 28.5 ± 1.3% |
| 7 | 10.6 ± 0.5% | 27.1 ± 1.2% | 27.3 ± 1.2% |
| 8 | 27.5 ± 1.3% | 28.2 ± 1.3% | 28.4 ± 1.3% |

## Native contribution captured

| K | Teacher top-K weight mass | Baseline predicted-set mass | Raw-score child mass | Native-weight child mass |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24.4% | 12.3% | 15.7% | 16.4% |
| 2 | 41.3% | 23.5% | 25.8% | 27.1% |
| 3 | 54.9% | 33.9% | 33.8% | 35.5% |
| 4 | 66.2% | 43.5% | 40.4% | 42.4% |
| 5 | 76.1% | 52.4% | 46.2% | 48.4% |
| 6 | 84.9% | 60.6% | 51.1% | 53.5% |
| 7 | 92.8% | 68.0% | 55.5% | 57.9% |
| 8 | 100.0% | 73.9% | 59.2% | 61.7% |

The hierarchy averaged 91 parent clusters per layer; parent noise was 50.0%. It produced 97.3 final prototypes per layer for raw-score children and 104.7 for native-weight children. Child noise was 59.4% and 67.9%, respectively. Parent noise was retained as a region, and sufficiently large child-noise groups were retained, so fallback prototypes could be broad.

The raw-score child clusters had higher weight L1 than native-weight children at every K:

| K | Raw-score child weight L1 | Native-weight child weight L1 |
| ---: | ---: | ---: |
| 1 | 1.060 ± 0.033 | 1.019 ± 0.034 |
| 2 | 1.019 ± 0.032 | 0.971 ± 0.029 |
| 3 | 0.989 ± 0.029 | 0.939 ± 0.028 |
| 4 | 0.968 ± 0.030 | 0.918 ± 0.027 |
| 5 | 0.949 ± 0.029 | 0.901 ± 0.026 |
| 6 | 0.937 ± 0.030 | 0.889 ± 0.027 |
| 7 | 0.929 ± 0.029 | 0.883 ± 0.025 |
| 8 | 0.921 ± 0.029 | 0.875 ± 0.026 |

In both cases L1 declines as K grows because the predicted distribution is evaluated over a larger active set.

## Layer-level variation across the three splits

The following ranges are computed from each layer’s mean over seeds 23, 42, and 71. They show the weakest and strongest layer-level matched-K recall for selected K values and the full range of top-8 contributor coverage across layers.

| Child clustering | K | Lowest layer recall | Highest layer recall | Top-8 coverage range |
| --- | ---: | ---: | ---: | ---: |
| Raw scores | 1 | 37.8% (layer 36) | 85.3% (layer 28) | 56.3%–89.9% |
| Native weights | 1 | 37.4% (layer 27) | 84.6% (layer 28) | 57.0%–90.1% |
| Raw scores | 4 | 44.8% (layer 36) | 62.4% (layer 0) | 54.1%–73.0% |
| Native weights | 4 | 45.0% (layer 36) | 70.1% (layer 0) | 54.8%–81.2% |
| Raw scores | 8 | 48.7% (layer 35) | 66.1% (layer 0) | 48.7%–66.1% |
| Native weights | 8 | 50.1% (layer 35) | 74.2% (layer 0) | 50.1%–74.2% |

These ranges are layer means, not confidence intervals. The gap between the weakest and strongest layers is large, especially for top-1 retrieval. At K=4 and K=8, layer 0 is the strongest and layers 35–36 are weakest in both variants. A favorable model-wide mean therefore does not describe every layer equally well.

## Metric definitions and scope

At each K, the teacher target is the K highest-weight experts among its captured native top eight. Matched-K recall divides the predicted/teacher overlap by K; exact set is the fraction of states with identical K-sized sets. Top-8 coverage keeps the predicted pool at eight and counts whether the teacher’s top-K contributors are included. Since the prediction pool is fixed, coverage at K=8 equals K=8 recall.

Teacher top-K weight mass is the share of native top-8 weight assigned to its K highest-weight IDs. Predicted-set native mass is the native top-8 weight represented by the predicted IDs. Hierarchical weight L1 compares top-K-renormalized predicted and reference weights. For all three, zero L1 is perfect and two is the maximum for normalized distributions. Values ± are standard deviations across topic-group splits, not token-level confidence intervals.

The experiment compares selection and contribution-weight proxies only. The child cluster’s mean native weight vector is derived from teacher outputs; it is not evidence that prototype-derived embeddings for independent community experts will share a calibrated scale.

## Interpretation and limits

Native-weight children improved low-K ranking and exact-set agreement relative to raw-score children, but neither preserved the baseline’s top-8 candidate coverage. At K=8, native-weight child recall was 58.1% versus 69.8% for the per-expert baseline. The results are offline router metrics only. They do not establish generated-quality impact, behavior with contributed experts, live loading, registry updates, or cache performance.
