# PCA dimensionality sweep for HDBSCAN prototypes

**Recorded completion time:** 2026-10-07 19:09:14 EDT. This filename time is based on the final sweep summary’s local modification time; an exact run start time was not recorded.

## Objective and protocol

This experiment estimated a practical PCA dimension for HDBSCAN-based prototype derivation. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0. All 40 layers were evaluated on 192 prompt variants from 48 source topic groups. Each split fitted on 38 topic groups and held out 10; repeated runs used seeds 23, 42, and 71.

PCA was fitted on training hidden states only. HDBSCAN clustered the projected states with min_cluster_size 32, min_samples 5, and EOM. Prototype vectors remained means in the original 2,048-D hidden space. Euclidean scoring was used. One broad sweep used seed 23; the confirmation comparison repeated 128, 256, 384, and 512 dimensions across three topic-group splits.

## Broad one-split sweep

| PCA dimensions | Explained variance | Top-8 recall | Exact set | Calibrated weight L1 | Prototypes/layer | Noise |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 28.9% | 70.08% | 29.28% | 0.8189 | 876.1 | 45.73% |
| 32 | 39.4% | 70.12% | 29.28% | 0.8179 | 901.6 | 44.99% |
| 64 | 51.7% | 70.20% | 29.15% | 0.8143 | 944.5 | 44.79% |
| 128 | 65.1% | 70.35% | 29.05% | 0.8115 | 960.7 | 46.80% |
| 192 | 73.0% | 70.56% | 28.86% | 0.8079 | 947.8 | 49.75% |
| 256 | 78.4% | 70.69% | 28.87% | 0.8057 | 937.0 | 51.87% |
| 384 | 85.8% | 70.78% | 28.88% | 0.8040 | 922.9 | 55.18% |
| 512 | 90.4% | 70.79% | 28.83% | 0.8037 | 912.0 | 57.73% |
| 1,024 | 98.0% | 70.72% | 28.76% | 0.8048 | 897.6 | 60.99% |
| 2,048 | 100.0% | 70.67% | 28.76% | 0.8058 | 898.1 | 61.41% |

## Three-split confirmation

Values are macro-averages across 40 layer scores. ± is standard deviation across the three topic-group seeds, not token-level uncertainty.

| PCA dimensions | Top-8 recall, mean ± split SD | Exact set, mean ± split SD | Weight L1, mean ± split SD | Prototypes/layer | Explained variance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 69.51 ± 1.17% | 27.68 ± 1.35% | 0.8242 ± 0.0181 | 958.9 | 65.4% |
| 256 | 69.75 ± 1.24% | 27.54 ± 1.28% | 0.8200 ± 0.0195 | 935.9 | 78.7% |
| 384 | 69.83 ± 1.22% | 27.55 ± 1.28% | 0.8187 ± 0.0191 | 919.4 | 86.0% |
| 512 | 69.83 ± 1.23% | 27.50 ± 1.30% | 0.8187 ± 0.0194 | 909.5 | 90.6% |

## Interpretation and limits

The results form a broad quality plateau. 384-D and 512-D tied to displayed precision across the three splits; 384-D is a reasonable working candidate, not a universal optimum. Its 0.08-point recall difference from 256-D is smaller than split-to-split variation. Fitting time was not benchmarked systematically, and substantial HDBSCAN noise was handled with residual/fallback rules. Only one checkpoint and three topic-group seeds were evaluated.

