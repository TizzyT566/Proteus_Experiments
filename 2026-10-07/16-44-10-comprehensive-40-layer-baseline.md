# Comprehensive 40-layer fixed-prototype comparison

**Recorded completion time:** 2026-10-07 16:44:10 EDT. This filename time is based on the final result summary’s local modification time; an exact run start time was not recorded.

## Objective and protocol

This evaluation expanded prototype routing to every MoE layer and compared clustering and similarity choices against the native router. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, a post-trained proxy checkpoint.

The model trace has 40 MoE layers, 256 experts per layer, and 2,048-dimensional hidden states. The reference checkpoint was captured in BF16 without quantization using GPU/CPU/disk offload. The corpus contained 192 prompt variants from 48 source topics. Variants from the same topic stayed together; one split fit on 38 topics and held out 10, yielding 8,592 training and 2,172 held-out router-input states per layer. Fixed cosine or Euclidean k-means used up to four prototypes per expert and a minimum cluster size of 32. For an expert, the router score was the maximum similarity over its prototypes. A per-layer softmax temperature was fit on training tokens only. The results below are macro-averages over layers for this single split.

## Results

| Clustering | Prototype scoring | Top-8 recall | Exact set | Calibrated weight L1 | Score Spearman | Reference top-8 supported |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cosine | Cosine | 63.7% | 8.7% | 0.935 | 0.449 | 99.7% |
| Cosine | Dot product | 63.7% | 8.7% | 0.935 | 0.449 | 99.7% |
| Cosine | Euclidean | 63.7% | 8.7% | 0.935 | 0.449 | 99.7% |
| Euclidean | Cosine | 63.2% | 7.9% | 0.941 | 0.448 | 99.7% |
| Euclidean | Raw dot product | 38.7% | 1.3% | 1.366 | 0.407 | 99.7% |
| Euclidean | Euclidean | 64.6% | 10.5% | 0.918 | 0.422 | 99.7% |

Euclidean clustering with Euclidean scoring was strongest among these tested settings, by about 0.9 percentage points of recall over cosine clustering/scoring. This is a small single-split difference, not a significance result. Cosine-normalized centroids produced nearly identical rankings under cosine, dot-product, and Euclidean scoring. Raw dot products on unnormalized Euclidean centroids ranked poorly, indicating that centroid norm can distort scores.

The fixed Euclidean baseline had 605.5 prototypes per layer on average. Its 64-D HDBSCAN and full-dimensional HDBSCAN comparisons are separate experiments.

## Layer-resolved fixed cosine results

This table expands the aggregate fixed cosine result. Support is the fraction of held-out states whose native top-8 set contains only experts represented by training prototypes. Recall and exact-set results use all held-out states.

| Layer | Experts covered | Top-8 supported | Recall | Exact set | Calibrated weight L1 | Score Spearman |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 243/256 (94.9%) | 99.4% | 80.6% | 33.3% | 0.692 | 0.308 |
| 1 | 240/256 (93.8%) | 99.9% | 68.3% | 14.0% | 0.880 | 0.380 |
| 2 | 252/256 (98.4%) | 100.0% | 62.3% | 13.1% | 1.033 | 0.505 |
| 3 | 244/256 (95.3%) | 99.8% | 63.9% | 13.3% | 0.965 | 0.344 |
| 4 | 242/256 (94.5%) | 99.1% | 70.6% | 11.2% | 0.886 | 0.374 |
| 5 | 243/256 (94.9%) | 99.6% | 63.2% | 6.2% | 0.958 | 0.452 |
| 6 | 223/256 (87.1%) | 99.8% | 62.9% | 9.6% | 0.959 | 0.453 |
| 7 | 231/256 (90.2%) | 99.6% | 62.8% | 6.0% | 0.946 | 0.395 |
| 8 | 225/256 (87.9%) | 96.9% | 61.3% | 4.6% | 0.966 | 0.408 |
| 9 | 235/256 (91.8%) | 100.0% | 61.0% | 5.1% | 1.011 | 0.431 |
| 10 | 234/256 (91.4%) | 99.9% | 64.1% | 8.9% | 0.916 | 0.484 |
| 11 | 234/256 (91.4%) | 99.6% | 66.4% | 11.8% | 0.894 | 0.502 |
| 12 | 249/256 (97.3%) | 100.0% | 61.7% | 5.4% | 1.000 | 0.491 |
| 13 | 232/256 (90.6%) | 99.9% | 59.9% | 4.9% | 1.016 | 0.460 |
| 14 | 244/256 (95.3%) | 99.8% | 62.0% | 8.5% | 0.935 | 0.489 |
| 15 | 245/256 (95.7%) | 99.6% | 61.2% | 8.2% | 0.943 | 0.455 |
| 16 | 235/256 (91.8%) | 100.0% | 62.5% | 5.7% | 0.983 | 0.437 |
| 17 | 233/256 (91.0%) | 98.8% | 59.0% | 6.7% | 1.011 | 0.411 |
| 18 | 238/256 (93.0%) | 99.8% | 58.5% | 5.2% | 1.006 | 0.438 |
| 19 | 236/256 (92.2%) | 100.0% | 59.6% | 7.5% | 0.974 | 0.469 |
| 20 | 238/256 (93.0%) | 99.0% | 56.2% | 5.7% | 1.051 | 0.386 |
| 21 | 237/256 (92.6%) | 99.7% | 58.3% | 4.2% | 1.098 | 0.415 |
| 22 | 240/256 (93.8%) | 100.0% | 62.8% | 5.2% | 0.936 | 0.422 |
| 23 | 247/256 (96.5%) | 100.0% | 64.3% | 10.7% | 0.924 | 0.493 |
| 24 | 248/256 (96.9%) | 99.8% | 60.1% | 5.7% | 1.051 | 0.446 |
| 25 | 240/256 (93.8%) | 100.0% | 60.0% | 5.1% | 1.037 | 0.421 |
| 26 | 249/256 (97.3%) | 100.0% | 59.9% | 6.3% | 0.987 | 0.468 |
| 27 | 250/256 (97.7%) | 99.8% | 58.8% | 4.9% | 0.988 | 0.497 |
| 28 | 231/256 (90.2%) | 99.9% | 61.9% | 3.3% | 1.035 | 0.475 |
| 29 | 233/256 (91.0%) | 99.6% | 61.2% | 7.6% | 0.959 | 0.507 |
| 30 | 234/256 (91.4%) | 100.0% | 62.8% | 9.3% | 0.930 | 0.484 |
| 31 | 229/256 (89.5%) | 100.0% | 59.6% | 3.5% | 0.974 | 0.503 |
| 32 | 221/256 (86.3%) | 99.8% | 70.4% | 11.0% | 0.786 | 0.469 |
| 33 | 229/256 (89.5%) | 99.5% | 67.5% | 5.2% | 0.894 | 0.430 |
| 34 | 212/256 (82.8%) | 99.6% | 73.6% | 12.8% | 0.727 | 0.397 |
| 35 | 229/256 (89.5%) | 100.0% | 63.4% | 12.1% | 0.884 | 0.535 |
| 36 | 222/256 (86.7%) | 99.7% | 66.3% | 13.4% | 0.824 | 0.517 |
| 37 | 221/256 (86.3%) | 99.6% | 66.7% | 9.1% | 0.794 | 0.533 |
| 38 | 208/256 (81.2%) | 99.4% | 66.7% | 8.3% | 0.879 | 0.463 |
| 39 | 211/256 (82.4%) | 99.9% | 73.7% | 15.3% | 0.672 | 0.399 |

## Per-layer comparison of fixed similarity choices

Top-8 recall for the cosine baseline, the strongest tested fixed Euclidean condition, and raw dot-product scoring on Euclidean centroids. The difference column is Euclidean/Euclidean minus cosine/cosine in percentage points.

| Layer | Cosine/cosine | Euclidean/Euclidean | Difference | Euclidean/raw dot |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 80.6% | 78.9% | -1.7 pp | 61.0% |
| 1 | 68.3% | 65.8% | -2.5 pp | 46.3% |
| 2 | 62.3% | 58.6% | -3.7 pp | 37.8% |
| 3 | 63.9% | 61.3% | -2.6 pp | 37.4% |
| 4 | 70.6% | 70.6% | 0.0 pp | 45.0% |
| 5 | 63.2% | 62.6% | -0.6 pp | 39.9% |
| 6 | 62.9% | 65.0% | +2.2 pp | 37.0% |
| 7 | 62.8% | 64.6% | +1.8 pp | 35.7% |
| 8 | 61.3% | 60.3% | -1.0 pp | 32.0% |
| 9 | 61.0% | 62.1% | +1.1 pp | 35.1% |
| 10 | 64.1% | 67.2% | +3.1 pp | 38.3% |
| 11 | 66.4% | 68.0% | +1.7 pp | 41.7% |
| 12 | 61.7% | 65.4% | +3.7 pp | 35.9% |
| 13 | 59.9% | 61.6% | +1.7 pp | 36.0% |
| 14 | 62.0% | 61.4% | -0.6 pp | 37.5% |
| 15 | 61.2% | 62.0% | +0.8 pp | 36.3% |
| 16 | 62.5% | 63.2% | +0.8 pp | 33.3% |
| 17 | 59.0% | 61.2% | +2.2 pp | 34.6% |
| 18 | 58.5% | 60.7% | +2.2 pp | 36.9% |
| 19 | 59.6% | 60.8% | +1.2 pp | 33.5% |
| 20 | 56.2% | 57.6% | +1.4 pp | 32.3% |
| 21 | 58.3% | 59.7% | +1.4 pp | 34.4% |
| 22 | 62.8% | 64.7% | +1.8 pp | 36.3% |
| 23 | 64.3% | 66.3% | +2.0 pp | 39.2% |
| 24 | 60.1% | 60.8% | +0.7 pp | 35.2% |
| 25 | 60.0% | 62.1% | +2.2 pp | 36.4% |
| 26 | 59.9% | 60.9% | +1.0 pp | 34.7% |
| 27 | 58.8% | 61.2% | +2.4 pp | 33.6% |
| 28 | 61.9% | 63.4% | +1.6 pp | 35.4% |
| 29 | 61.2% | 63.5% | +2.3 pp | 38.6% |
| 30 | 62.8% | 65.0% | +2.2 pp | 40.1% |
| 31 | 59.6% | 61.8% | +2.2 pp | 34.1% |
| 32 | 70.4% | 70.5% | +0.1 pp | 42.3% |
| 33 | 67.5% | 69.0% | +1.5 pp | 45.7% |
| 34 | 73.6% | 75.3% | +1.7 pp | 49.1% |
| 35 | 63.4% | 63.8% | +0.4 pp | 40.3% |
| 36 | 66.3% | 65.8% | -0.5 pp | 41.6% |
| 37 | 66.7% | 68.2% | +1.5 pp | 40.1% |
| 38 | 66.7% | 67.0% | +0.3 pp | 39.6% |
| 39 | 73.7% | 74.9% | +1.1 pp | 47.8% |

## Scoring and metric definitions

For each layer and expert, training states associated with that native expert were clustered into at most four centroids. The held-out expert score was the best similarity over that expert’s centroids. The score table compares the clustering metric with the separate metric used to rank centroids. On cosine centroids, the tested score formulas induce the same expert ordering; on Euclidean centroids, raw dot product behaves differently because it includes centroid norm.

Top-8 recall is the number of native top-8 experts recovered divided by eight, averaged over held-out states. Exact set is the fraction of states where predicted and native top-8 sets match exactly. Calibrated weight L1 is the sum of absolute differences between dense 256-expert distributions after fitting a temperature on training tokens; zero is perfect and two is the maximum for normalized distributions. Score Spearman measures rank correlation between prototype scores and native gate scores. “Top-8 supported” is a diagnostic: it counts held-out states whose entire native top-8 can be represented by experts with fitted prototypes. The main recall and L1 averages include all held-out states.

Uniform random selection among 256 experts has an expected top-8 recall of 3.125%. The results are substantially above this reference, but candidate recovery remains different from reproducing the native score distribution.

## Interpretation and limits

The fixed-count prototype setup recovered useful candidates but did not reproduce the complete native top-8 set reliably. Exact-set agreement averaged 8.7% for the cosine setup and 10.5% for the strongest Euclidean setup. The held-out split contained only 10 independent source topics; prompt variants add trace volume but not independent topic coverage. These offline routing metrics do not measure generated quality.
