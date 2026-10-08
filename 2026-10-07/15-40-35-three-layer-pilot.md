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

## Detailed reading of the pilot

The expected top-8 recall for uniform selection among 256 experts is 8/256 = 3.125%. The measured recalls are well above that reference, but recall alone does not indicate whether the selected experts receive appropriate weights. Layer 0 recovered about 6.0 of the teacher’s eight experts per state; layer 20 recovered about 3.9; layer 39 recovered about 5.6.

Coverage and supported-state rate diagnose separate failure modes. Expert coverage is how many distinct native experts received at least one prototype during fitting. Top-8 support is the held-out state fraction where all eight native choices had a corresponding trained prototype. Layer 39 had the lowest expert coverage (193/256) but 98.1% supported states, while layer 20 had 230/256 experts represented and 97.8% support. Thus, expert coverage alone did not explain the weak layer-20 recall.

The best-of-sweep readout recalls (55.8%, 48.6%, and 60.6%) were not a held-out model-selection estimate: the same held-out prompts informed the exploratory sweep. They should not be compared to the fixed baseline as an unbiased improvement.

## Interpretation and limits

Prototype retrieval exceeded random top-8 selection, but performance varied substantially by layer. Layer 20 was the main failure point. The held-out sample was only 10 prompts, and prototype coverage ranged from 75% to 93% of experts. This pilot did not establish reliable full-depth behavior or generation-quality impact.

Recall is the fraction of native selected experts recovered per token. Weight L1 compares dense 256-expert routing distributions; zero is perfect and two is the maximum.
