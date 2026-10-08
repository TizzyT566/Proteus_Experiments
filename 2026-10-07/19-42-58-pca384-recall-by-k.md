# PCA-384 prototype recall by active K

**Recorded completion time:** 2026-10-07 19:42:58 EDT. This filename time is based on the aggregate result timestamp; an exact run start time was not recorded.

## Objective and protocol

This analysis measured how PCA-384 HDBSCAN prototypes recovered native router contributors when the predicted active count varied from K=1 to K=8. It reused the 40-layer prototype evaluation and ran across topic-group seeds 23, 42, and 71. HDBSCAN used min_cluster_size 32, min_samples 5, and EOM; prototype scores used Euclidean max-prototype scoring.

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

## Interpretation and limits

The matched-K recall rises with larger K for this baseline, but exact-set rates remain low for K=2 through K=6. At K=8, 69.83% recall means about 5.6 of the eight reference experts are recovered on average. The prediction’s K-set captures 73.95% of native top-8 weight at K=8. These are router-proxy metrics, not generation-quality measurements.

