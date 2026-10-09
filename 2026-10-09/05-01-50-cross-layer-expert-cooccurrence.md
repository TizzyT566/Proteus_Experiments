# Cross-layer expert co-occurrence

**Checkpoint:** `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`  
**Artifact time:** 2026-10-09 05:01:50 America/New_York  
**Source:** native top-eight expert IDs from 40 layers and 22,296 aligned token positions in the 384-prompt expert-profile corpus. Training and validation positions are both included.

## Definition

For every token position and every ordered pair of layers (L_i < L_j), the analysis paired each of the eight selected experts at (L_i) with each of the eight selected experts at (L_j). Thus each layer pair has 64 cross-layer pair observations per token, or 1,426,944 across the corpus. It counts whether the two experts were selected for the same token as it passed through those layers; “then” here means later in the same model forward pass, not a later generated token.

Layer row alignment was checked by exact equality of `group_ids` and `sequence_ids` across the 40 trace files. Expert IDs are layer-local. A layer-aware pair includes both layer IDs and expert IDs; expert 52 in layer 2 is distinct from expert 52 in layer 3.

## Overall counts

There are 780 distinct layer pairs. For each layer pair, 256 × 256 = 65,536 expert-ID pairs are possible, or 51,118,080 layer-aware pair types across all layer pairs. The corpus observed **18,725,437** (36.63%) of these possible types across **1,113,016,320** pair observations.

| Layer distance | Layer pairs | Distinct observed pairs per layer pair, mean (range) | Most frequent pair count per layer pair, mean | Mean token share of the most frequent pair |
|---:|---:|---:|---:|---:|
| 1 | 39 | 22,766 (18,021–28,914) | 9,840 | 44.1% |
| 2 | 38 | 23,077 (18,610–26,944) | 9,412 | 42.2% |
| 10 | 30 | 24,111 (21,177–29,431) | 9,606 | 43.1% |
| 20 | 20 | 24,736 (19,420–33,343) | 9,101 | 40.8% |
| 39 | 1 | 20,542 | 2,502 | 11.2% |

The most frequent pair across the measured layer pairs was layer 29 expert 130 with layer 39 expert 209: it occurred together at 21,621 of 22,296 positions (96.97%). This is a raw count, not evidence of a special functional relationship; both experts may simply be selected often in this corpus.

## Example IDs

Under one-based layer numbering, layer 1 expert 52 and layer 3 expert 5 were selected together at **8** positions. Layer 2 expert 52 and layer 3 expert 5 were selected together at **0** positions. If the intended example is a three-layer chain with an exact expert required at layer 2, that is a three-way query and needs the layer-2 expert ID; the pair result above does not constrain which expert was selected in the middle layer.

## Interpretation and limits

Raw co-selection counts show how often route decisions overlap on the same profiled token. They are affected by each expert's marginal selection frequency and by the repeated instruction frames in the corpus. To identify unusually strong associations, compare observed counts with an independence baseline (or calculate lift/PMI). To identify complementarity or usefulness, measure counterfactual loss changes. Neither causal interaction nor deployment behavior is established by these pair counts.

## Artifacts

Reproduce with `experiments/analyze_cross_layer_cooccurrence.py`. The complete 780 layer-pair summaries and expert-52-to-expert-5 counts are in `results/cross_layer_expert_cooccurrence_qwen36.json`.
