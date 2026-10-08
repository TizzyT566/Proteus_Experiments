# Three-layer prototype-router pilot

**Recorded completion time:** 2026-10-07 15:40:35 EDT. This filename time is based on the final result summary’s local modification time; an exact run start time was not recorded.

## Objective and setup

This initial feasibility run tested whether hidden-state centroids could recover the frozen native router’s selected experts. It used Qwen/Qwen3.6-35B-A3B, revision 995ad96eacd98c81ed38be0c5b274b04031597b0. The checkpoint is post-trained and is a proxy for a Base checkpoint.

The capture used BF16 weights without quantization and GPU/CPU/disk offload. It recorded 1,851 router-input states for each of layers 0, 20, and 39, including all 256 router logits, selected expert IDs, and native top-8 weights. The corpus had 48 varied prompts, with 38 used for fitting/calibration and 10 held out by prompt. Every comparison used K=8.

## Fixed-centroid results

The baseline used cosine k-means, up to four prototypes per expert, minimum cluster size 32, cosine similarity, max pooling across an expert’s prototypes, and one positive softmax temperature per layer fitted on training prompts.

| Layer | Experts represented | Held-out top-8 support | Top-8 recall | Exact top-8 set | Weight L1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 237/256 | 99.2% | 75.3% | 15.3% | 0.762 |
| 20 | 230/256 | 97.8% | 49.0% | 2.8% | 1.161 |
| 39 | 193/256 | 98.1% | 70.0% | 5.3% | 0.725 |

Two score mappings were also tested: a per-expert affine fit from best prototype similarity and a ridge readout over prototype similarities. Neither repaired the weak middle-layer result. The best recall observed in the broader exploratory readout sweep was 55.8%, 48.6%, and 60.6% for layers 0, 20, and 39; these are best-of-sweep values measured on the same held-out prompts and are optimistic diagnostics.

## Interpretation and limits

Prototype retrieval exceeded random top-8 selection, but performance varied substantially by layer. Layer 20 was the main failure point. The held-out sample was only 10 prompts, and prototype coverage ranged from 75% to 93% of experts. This pilot did not establish reliable full-depth behavior or generation-quality impact.

Recall is the fraction of native selected experts recovered per token. Weight L1 compares dense 256-expert routing distributions; zero is perfect and two is the maximum.

