# PCA-384 prototype recall by active K

**Recorded completion time:** 2026-10-07 19:42:58 EDT. This filename time is based on the aggregate result timestamp; an exact run start time was not recorded.

## Objective and protocol

This analysis measured how PCA-384 HDBSCAN prototypes recovered native router contributors when the predicted active count varied from K=1 to K=8. It reused the 40-layer, 256-expert-per-layer prototype evaluation on 192 prompt variants from 48 source topics, with 38 topics for fitting and 10 held out in each split. Each layer had 8,592 training and 2,172 evaluation states. The analysis ran across topic-group seeds 23, 42, and 71. HDBSCAN used min_cluster_size 32, min_samples 5, and EOM; prototype scores used Euclidean max-prototype scoring.

For K below eight, the teacher target was the K native experts with the largest captured gate weights. Reported values are macro-averages over layers. Split SD is across the three topic-group splits.

## Results

| K | Matched-K recall, mean ± split SD | Teacher top-K covered in predicted top 8, mean ± SD | Exact set, mean ± split SD | Teacher top-K weight mass | Predicted K-set native mass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.90 ± 0.06% | 83.29 ± 1.38% | 16.90 ± 0.06% | 24.37% | 12.30% |
| 2 | 29.26 ± 0.37% | 82.31 ± 1.31% | 6.25 ± 0.14% | 41.30% | 23.53% |
| 3 | 39.40 ± 0.41% | 80.69 ± 1.29% | 3.33 ± 0.04% | 54.85% | 33.90% |
| 4 | 48.06 ± 0.62% | 78.79 ± 1.28% | 2.39 ± 0.10% | 66.25% | 43.46% |
| 5 | 55.24 ± 0.82% | 76.71 ± 1.29% | 2.35 ± 0.09% | 76.12% | 52.35% |
| 6 | 61.37 ± 1.00% | 74.59 ± 1.30% | 3.90 ± 0.08% | 84.86% | 60.56% |
| 7 | 66.36 ± 1.13% | 72.36 ± 1.26% | 10.58 ± 0.52% | 92.76% | 67.97% |
| 8 | 69.83 ± 1.22% | 69.83 ± 1.22% | 27.55 ± 1.28% | 100.00% | 73.95% |

Matched-K recall is overlap between the predicted K experts and the teacher’s K highest-weight experts, divided by K. It need not improve monotonically as K decreases: the predicted rank-1 expert can be missed while lower-weight teacher experts are recovered at larger K. Top-8-pool coverage holds the predicted candidate pool at eight and measures whether teacher contributors appear anywhere in it.

## Layer variation

The table reports the minimum and maximum matched-K recall across the 40 layer means, where each layer mean averages the three topic-group seeds. The final columns identify the layers attaining those endpoints.

| K | Lowest layer mean | Highest layer mean |
| ---: | ---: | ---: |
| 1 | 7.5% (layer 28) | 21.8% (layer 4) |
| 2 | 20.1% (layer 25) | 34.3% (layer 35) |
| 3 | 30.9% (layer 25) | 45.1% (layer 34) |
| 4 | 41.6% (layer 25) | 54.8% (layer 34) |
| 5 | 50.8% (layer 25) | 63.3% (layer 34) |
| 6 | 57.3% (layer 20) | 70.5% (layer 0) |
| 7 | 61.8% (layer 20) | 77.4% (layer 0) |
| 8 | 64.8% (layer 20) | 82.1% (layer 0) |

The model-wide mean masks substantial layer variation. Layer 20 is the weakest at K=6–8, while layer 0 is strongest at those K values. Layer 25 is weakest at K=2–5. For K=1, layer 4—not layer 0—has the strongest mean top-1 match.

## Metric details

The teacher’s top-K target is formed by sorting its native top-8 experts by their normalized gate weights; the captured expert ID array itself is unordered. Matched-K recall and exact-set rate compare predicted and teacher sets of the same size. Top-8-pool coverage instead uses a fixed prediction pool of eight while the teacher target varies from one to eight. The two metrics answer different retrieval questions.

Teacher top-K weight mass is the average normalized weight on those K teacher contributors. Predicted K-set native mass is the sum of native teacher weights assigned to the predicted K IDs, divided by total captured native top-8 mass. Therefore, at K=8 it is possible to recover 69.83% of the teacher expert IDs while capturing 73.95% of their total gate weight. All reported percentages are macro-averages across 40 layers; split SD reflects three topic-group seeds.

## Interpretation and limits

The matched-K recall rises with larger K for this baseline, but exact-set rates remain low for K=2 through K=6. At K=8, 69.83% recall means about 5.6 of the eight reference experts are recovered on average. The prediction’s K-set captures 73.95% of native top-8 weight at K=8. These are router-proxy metrics, not generation-quality measurements.
