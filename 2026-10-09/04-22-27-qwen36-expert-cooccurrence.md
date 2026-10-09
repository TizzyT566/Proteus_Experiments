# Qwen3.6 routed expert co-occurrence

**Checkpoint:** `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`  
**Artifact time:** 2026-10-09 04:22:27 America/New_York  
**Source:** native top-eight IDs from all 40 MoE layers and all 22,296 token positions in the 384-prompt expert-profile corpus. This includes training and validation prompts.

## Method

For each token and each subset size k from 2 to 8, the analysis enumerated every unordered k-subset of that token's eight selected experts. It counted distinct subsets and their frequencies separately for each layer. Expert IDs are layer-local: expert 17 at layer 3 is a different module from expert 17 at layer 20.

Each token contributes 28 pair instances, 56 triples, 70 four-expert groups, 56 five-expert groups, 28 six-expert groups, 8 seven-expert groups, and one complete top-eight set. “Distinct observed” counts different combinations seen at least once. “Singleton” is the share of those observed combinations seen only once in that layer's corpus.

## Results across layers

| Group size | Possible combinations from 256 | Distinct observed per layer, mean (range) | Median occurrences per observed group, mean across layers | Observed groups occurring once, mean |
|---:|---:|---:|---:|---:|
| 2 | 32,640 | 10,587 (6,886–13,739) | 9.55 | 15.1% |
| 3 | 2,763,520 | 77,492 (38,037–104,163) | 4.10 | 27.6% |
| 4 | 174,792,640 | 197,417 (87,173–260,938) | 2.45 | 37.4% |
| 5 | 8,809,549,056 | 242,334 (103,104–318,289) | 1.90 | 45.6% |
| 6 | 368,532,802,176 | 159,891 (67,391–205,766) | 1.33 | 52.5% |
| 7 | 13,161,885,792,000 | 55,387 (23,459–69,690) | 1.10 | 58.5% |
| 8 | 409,663,695,276,000 | 7,991 (3,430–9,860) | 1.08 | 63.5% |

For example, layer 20 observed 11,531 different expert pairs. Its most frequent pair, experts 191 and 224, appeared together in 5,165 of 22,296 positions (23.2%). Across layers, the most frequent pair appeared an average of 7,184 times, with the per-layer maximum ranging from 2,128 to 14,965 positions.

Complete per-layer frequency summaries, including the most common group for every size, are in the JSON artifact.

## Interpretation and limits

Pair reuse is substantial: on average, 98.3% of pair instances are repeats after counting each distinct pair's first occurrence. Larger sets become much less stable; an average of 63.5% of observed eight-expert sets occur only once. This points to recurring small pair relationships alongside much more varied full top-eight teams.

These are observations from one checkpoint and one prompt corpus with repeated instruction frames. They show co-selection, not causal cooperation, interchangeability, or marginal usefulness. Repeated prompt prefixes contribute multiple token positions, and the analysis pools training and validation positions for usage statistics. Expert combinations can differ substantially by layer.

## Artifacts

The analysis is reproducible with `experiments/analyze_expert_cooccurrence.py`. Full counts are saved in `results/expert_cooccurrence_qwen36.json`.
