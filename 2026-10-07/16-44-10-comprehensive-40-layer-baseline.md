# Comprehensive 40-layer fixed-prototype comparison

**Recorded completion time:** 2026-10-07 16:44:10 EDT. This filename time is based on the final result summary’s local modification time; an exact run start time was not recorded.

## Objective and protocol

This evaluation expanded prototype routing to every MoE layer and compared clustering and similarity choices against the native router. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, a post-trained proxy checkpoint.

The corpus contained 192 prompt variants from 48 source topics. Variants from the same topic stayed together; one split fit on 38 topics and held out 10. All 40 layers were evaluated. Fixed cosine or Euclidean k-means used up to four prototypes per expert and a minimum cluster size of 32. A per-layer softmax temperature was fit on training tokens only. The results below are macro-averages over layers for this single split.

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

## Interpretation and limits

The fixed-count prototype setup recovered useful candidates but did not reproduce the complete native top-8 set reliably. Exact-set agreement averaged 8.7% for the cosine setup and 10.5% for the strongest Euclidean setup. The held-out split contained only 10 independent source topics; prompt variants add trace volume but not independent topic coverage. These offline routing metrics do not measure generated quality.

