# Three-layer HDBSCAN prototype pilot

**Recorded completion time:** 2026-10-07 17:07:52 EDT. This filename time is based on the final result artifact’s local modification time; an exact run start time was not recorded.

## Objective and method

This follow-up tested variable-count density clustering on the same three representative layers as the initial pilot: 0, 20, and 39. For each layer and native expert, HDBSCAN discovered clusters in a train-only 64-dimensional PCA projection; prototype vectors were means in the original hidden-state space.

The run used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0, with 192 prompt variants from 48 source topics; 38 topics were used for fitting and 10 held out. This gave 8,592 training and 2,172 held-out router-input states per layer. HDBSCAN used min_cluster_size 32, min_samples 5, and EOM selection. Large noise sets received one residual prototype; small noise sets were assigned to their nearest discovered cluster, and experts with no clusters received a global fallback. Calibration was fitted on training tokens.

## Results

| Layer | Prototypes | Mean clusters/expert | Noise fraction | Top-8 supported | Cosine recall | Dot-product recall | Euclidean recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 960 | 3.28 | 29.9% | 99.4% | 83.1% | 72.4% | 83.2% |
| 20 | 950 | 3.24 | 41.0% | 99.0% | 64.6% | 53.6% | 64.0% |
| 39 | 982 | 3.81 | 50.3% | 99.9% | 77.1% | 66.7% | 77.5% |

## Per-layer scoring and density behavior

Cosine and Euclidean scoring were effectively tied on these sampled layers: their recall differences were +0.1, -0.6, and +0.4 points for layers 0, 20, and 39. Raw dot-product scoring was lower by 10.7, 11.0, and 10.8 points, respectively. That pattern is consistent with raw centroid magnitude changing rankings even when direction is informative.

HDBSCAN produced roughly 950–982 final prototypes per layer, or about 3.2–3.8 discovered density clusters per represented expert on average. Noise increased from 29.9% at layer 0 to 50.3% at layer 39. The residual/fallback-centroid policy matters because a substantial share of training states did not belong to a density cluster; these results evaluate the combined cluster-plus-fallback procedure, not density clusters alone.

All three layers had at least 99.0% held-out top-8 support. That support statistic means the teacher’s full top-8 set was representable by fitted experts for that fraction of states; it is not the fraction of experts covered, nor a measure of correct ranking.

## Metric definitions

Top-8 recall is overlap of predicted and native top-8 expert IDs divided by eight, averaged over held-out states. Exact set is the fraction of states with identical sets. Calibrated weight L1 compares dense native and predicted gate distributions after the training-only temperature fit; zero is perfect and two is the maximum for normalized distributions. “Top-8 supported” counts states where all teacher-selected experts had training prototypes available. It does not measure ranking accuracy.

The noise percentage is the share of training states HDBSCAN labeled as noise before residual/fallback handling. The prototype count includes the residual and fallback prototypes used to cover those states. Thus the routing metrics evaluate the complete fallback-enabled prototype bank.

## Interpretation and limits

The variable-count prototypes improved recall over the initial fixed-centroid pilot on all three sampled layers, particularly layer 20. This was a three-layer, single-split pilot and does not establish the full-depth result. Noise handling and the number of clusters depend on HDBSCAN settings; the cluster count was selected by density stability, not directly optimized for routing accuracy.
