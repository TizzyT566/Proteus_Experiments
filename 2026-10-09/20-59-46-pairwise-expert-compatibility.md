# Same-layer pairwise expert compatibility

**Run completed:** 2026-10-09 20:59:46 EDT (Windows local time; based on the final result artifact timestamp)  
**Model:** `Qwen/Qwen3.6-35B-A3B`  
**Checkpoint revision:** `995ad96eacd98c81ed38be0c5b274b04031597b0`  
**Runtime:** Windows, CUDA device `cuda:0` for the hidden-state predictor; pair statistics and greedy selection on CPU  
**Elapsed time:** 138.9 seconds for all 40 layers

**Measurement correction:** 2026-10-09 EDT. The original mixture-weight L1 omitted native weight on experts missing from the predicted set. The original aggregate values were 0.303063 for relevance only and 0.297479 for compatibility. Full dense L1 is 0.465919 and 0.456447, respectively. The correction reused the saved predictors, selected interaction strengths, topic split, and route temperatures; it did not refit predictors or calibration, retune strengths, or run generation. The experiment completion time above is unchanged.

## Question

Can same-layer expert-to-expert compatibility improve expert selection and mixture weighting when added to a hidden-state relevance score?

The test deliberately treated compatibility as an **interaction term**, not a replacement for hidden-state relevance. For hidden state `h`, the existing per-expert affine predictor supplies a relevance score `u_e(h)`. A learned compatibility score `C(e,f)` adjusts the set:

```text
choose a size-8 set S to approximately maximize:
    sum over e in S of u_e(h)
  + λ × sum over distinct pairs e,f in S of C(e,f)
```

The second term rewards a set whose members tended to be selected together beyond what their individual selection rates would predict. The selected experts' mixture logits then receive the average compatibility with the other selected experts before softmax weighting.

## Data and held-out protocol

The run used the saved router traces for 384 prompts grouped into 48 topic groups. Each of the 40 layers has 22,296 token positions, 256 experts, a 2,048D hidden state, and eight native selected experts. The outer split held out whole topic groups using seed 23: 38 groups (17,792 positions) for fitting and 10 groups (4,504 positions) for evaluation in each layer. The total held-out set contains 180,160 layer-token observations.

The interaction strength `λ` was selected separately for each layer using an inner topic-group split of the outer training data: 30 groups for fitting and eight for validation. The inner predictor was refit on the inner-fit rows using that layer's saved ridge penalty. The outer held-out groups were not used to fit the compatibility table or select `λ`.

The first provisional pass was discarded after detecting that reversed expert-ID slot order could double-count an unordered pair. The reported run counts each pair once per token using canonical `(min ID, max ID)` order. Same-expert diagonal entries are undefined for pair compatibility and are set to zero.

## Compatibility estimate and selection

For each layer, training top-eight selections produce expert marginals and unordered pair counts. For experts `e` and `f`, compatibility starts with the binary-selection log odds ratio:

```text
log(((n11 + 1)(n00 + 1)) / ((n10 + 1)(n01 + 1)))
```

where `n11` is the number of training positions selecting both, `n10` and `n01` select exactly one, and `n00` select neither. This discounts raw co-occurrence frequency by each expert's individual popularity. The log odds are shrunk toward zero by `expected_pair_count / (expected_pair_count + 20)`, then centered and scaled within the layer. The diagonal is zero.

At inference, all 256 affine relevance scores are normalized within each hidden state, then a greedy top-eight procedure chooses one expert at a time. At each step it adds the compatibility with experts already chosen. The per-layer `λ` was selected from `0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6`, maximizing inner-validation top-eight recall and using exact-set rate as the tie-breaker. Most layers selected a small value: 20 chose `0.025`, 17 chose `0.05`, two chose `0.1`, and one chose `0`.

No HDBSCAN or PCA was used. This experiment used exhaustive scoring of 256 experts, not ANN candidate retrieval.

## Held-out results

The baseline and compatibility route use the same affine hidden-state predictor. The set-recall and exact-set comparisons isolate the pairwise reranking more directly; mixture weights additionally reflect within-state unary normalization and each route's temperature calibration. Compatibility improved recall in 39 of 40 layers, improved exact-set rate in 24 of 40, and reduced full dense weight L1 in 36 of 40 (worse in four).

| Route | Top-8 mean recall | Exact top-8 set | Full dense mixture-weight L1 |
| --- | ---: | ---: | ---: |
| Affine relevance only | 0.787781 | 0.348174 | 0.465919 |
| Affine relevance + pair compatibility | **0.792791** | **0.349023** | **0.456447** |
| Change | +0.005010 | +0.000849 | -0.009472 |

The weight-L1 calibration here is conditional: for each predicted set, temperature is fit on training examples using the teacher's relative weights among experts shared with that predicted set. The corrected held-out metric is full dense L1 across all 256 expert coordinates: selected-coordinate error plus native weight on all missed teacher experts. Missing mass is the saved native weight sum minus native weight shared with predicted IDs; it is not assumed that the saved BF16 weights sum to exactly one. The corrected formula was checked against independently materialized dense distributions for all 80 layer/route combinations, with maximum absolute disagreement 5.05 × 10^-11. Identical, disjoint, and non-unit native-mass examples returned L1 of 0, 2, and 1.125. This experiment uses a different weight-temperature calibration from the earlier affine report; any cross-report comparison must account for that difference. The selection metrics reproduce the earlier affine baseline.

### Per-layer results

Layer IDs are zero-based. Full dense weight L1 is lower when predicted mixture weights are closer to the native weights; the table includes missed native probability mass.

| Layer | λ | Recall base | Recall + compat | Exact base | Exact + compat | Full L1 base | Full L1 + compat |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.025 | 0.8530 | 0.8554 | 0.4714 | 0.4702 | 0.3695 | 0.3587 |
| 1 | 0.050 | 0.8186 | 0.8258 | 0.4165 | 0.4123 | 0.4281 | 0.4114 |
| 2 | 0.050 | 0.7631 | 0.7717 | 0.3803 | 0.3797 | 0.5496 | 0.5298 |
| 3 | 0.000 | 0.7529 | 0.7529 | 0.3131 | 0.3131 | 0.5720 | 0.5561 |
| 4 | 0.025 | 0.8217 | 0.8261 | 0.4076 | 0.4141 | 0.4110 | 0.3967 |
| 5 | 0.050 | 0.7760 | 0.7841 | 0.3537 | 0.3541 | 0.4794 | 0.4653 |
| 6 | 0.050 | 0.7865 | 0.7927 | 0.3732 | 0.3761 | 0.4800 | 0.4661 |
| 7 | 0.050 | 0.7721 | 0.7795 | 0.3455 | 0.3535 | 0.5159 | 0.5008 |
| 8 | 0.025 | 0.7570 | 0.7628 | 0.3401 | 0.3384 | 0.5316 | 0.5227 |
| 9 | 0.025 | 0.7792 | 0.7834 | 0.3433 | 0.3452 | 0.4857 | 0.4741 |
| 10 | 0.100 | 0.8050 | 0.8089 | 0.3615 | 0.3646 | 0.4375 | 0.4308 |
| 11 | 0.050 | 0.8120 | 0.8171 | 0.3672 | 0.3681 | 0.4199 | 0.4087 |
| 12 | 0.050 | 0.7659 | 0.7741 | 0.3563 | 0.3479 | 0.5155 | 0.4959 |
| 13 | 0.025 | 0.7718 | 0.7777 | 0.3348 | 0.3368 | 0.4992 | 0.4871 |
| 14 | 0.050 | 0.7442 | 0.7506 | 0.3384 | 0.3379 | 0.5599 | 0.5472 |
| 15 | 0.100 | 0.7619 | 0.7710 | 0.3151 | 0.3073 | 0.5264 | 0.5020 |
| 16 | 0.025 | 0.7720 | 0.7749 | 0.3421 | 0.3419 | 0.5103 | 0.5088 |
| 17 | 0.025 | 0.7666 | 0.7707 | 0.3552 | 0.3586 | 0.5190 | 0.5092 |
| 18 | 0.025 | 0.7572 | 0.7602 | 0.3217 | 0.3224 | 0.5262 | 0.5183 |
| 19 | 0.050 | 0.7302 | 0.7362 | 0.3153 | 0.3115 | 0.5744 | 0.5593 |
| 20 | 0.050 | 0.7385 | 0.7464 | 0.3068 | 0.3075 | 0.5744 | 0.5706 |
| 21 | 0.025 | 0.7683 | 0.7715 | 0.3188 | 0.3222 | 0.5161 | 0.5227 |
| 22 | 0.025 | 0.8061 | 0.8072 | 0.3208 | 0.3219 | 0.4637 | 0.4660 |
| 23 | 0.050 | 0.7865 | 0.7944 | 0.3228 | 0.3286 | 0.4518 | 0.4446 |
| 24 | 0.025 | 0.7650 | 0.7693 | 0.3282 | 0.3293 | 0.5101 | 0.5029 |
| 25 | 0.025 | 0.7868 | 0.7900 | 0.3222 | 0.3253 | 0.4522 | 0.4591 |
| 26 | 0.025 | 0.7560 | 0.7597 | 0.3315 | 0.3333 | 0.5205 | 0.5154 |
| 27 | 0.050 | 0.7606 | 0.7672 | 0.2991 | 0.2975 | 0.5177 | 0.5031 |
| 28 | 0.025 | 0.7910 | 0.7943 | 0.3262 | 0.3246 | 0.4250 | 0.4261 |
| 29 | 0.025 | 0.7835 | 0.7883 | 0.3399 | 0.3417 | 0.4573 | 0.4460 |
| 30 | 0.025 | 0.7889 | 0.7930 | 0.3142 | 0.3139 | 0.4438 | 0.4369 |
| 31 | 0.050 | 0.7868 | 0.7905 | 0.3273 | 0.3253 | 0.4542 | 0.4494 |
| 32 | 0.050 | 0.8353 | 0.8421 | 0.3561 | 0.3546 | 0.3520 | 0.3392 |
| 33 | 0.025 | 0.8393 | 0.8438 | 0.3839 | 0.3914 | 0.3535 | 0.3343 |
| 34 | 0.025 | 0.8598 | 0.8640 | 0.3994 | 0.4021 | 0.3039 | 0.2984 |
| 35 | 0.025 | 0.7749 | 0.7797 | 0.3097 | 0.3111 | 0.4670 | 0.4585 |
| 36 | 0.050 | 0.8015 | 0.8071 | 0.3313 | 0.3297 | 0.4204 | 0.4120 |
| 37 | 0.050 | 0.8151 | 0.8207 | 0.3488 | 0.3526 | 0.3996 | 0.3928 |
| 38 | 0.025 | 0.8378 | 0.8400 | 0.3710 | 0.3737 | 0.3556 | 0.3505 |
| 39 | 0.050 | 0.8627 | 0.8668 | 0.4167 | 0.4212 | 0.2866 | 0.2806 |

## Interpretation

The captured traces contain a real but small same-layer compatibility signal. Pairwise reranking slightly raises recall and barely changes exact-set agreement. The combined route modestly improves mixture-weight L1 under the normalization and calibration used here. The selected interaction strength is small, which is consistent with compatibility refining a context-specific relevance score rather than replacing it.

**Methodology limitation:** the compatibility route standardizes unary scores within each query and fits its own temperature; the baseline uses unstandardized scores and a separately fitted temperature. Weight L1 improvement therefore cannot be attributed solely to the pair term. At layer 3, `λ = 0` leaves set recall and exact-set rate unchanged but weight L1 still changes from **0.571953 to 0.556069**. Set recall and exact-set rate measure the reranking contribution more directly than this joint weighting comparison.

This does **not** show that raw co-selection means two experts make each other more useful. The table imitates the native router's observed combinations; it cannot distinguish synergy from the native router's own biases. Measuring actual usefulness or causal cooperation requires comparing expert outputs or model loss under alternative expert sets.

The run also does not establish an arbitrary-size registry design. A dense compatibility table costs `N × N` values per layer: at 256 experts it is small, but it grows quadratically. The evaluated router still needs a hidden-state relevance signal for each candidate; compatibility alone has no input-context signal and can favor frequent combinations even when they are wrong for the current hidden state. No ANN latency, 10k-expert scaling, or generated-text quality was measured.

## Reproduction details

For each layer:

1. Partition prompt topic groups into outer train and held-out groups with seed 23 and train fraction 0.8.
2. Within outer training groups, split again with seed 1023 and fraction 0.8. Fit the affine predictor on inner-fit rows, estimate log-odds compatibility on those rows, and select `λ` on inner-validation recall (exact-set rate breaks ties).
3. Estimate a fresh compatibility table from all outer-training selected top-eight sets. Score held-out hidden states with the outer-training affine predictor and run greedy top-eight selection with the chosen `λ`.
4. Fit each route's mixture temperature using only outer-training rows, normalizing teacher weights over predicted/native shared experts. Evaluate held-out set recall, exact-set rate, and full-mixture L1.

The interaction shrinkage uses expected co-selection count plus 20 as the reliability denominator; log odds use Laplace smoothing of 1. The grid is `0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6`. Every reported aggregate is weighted over the same 4,504 held-out token positions per layer. This is one checkpoint and one topic split; no repeated-seed confidence interval was computed.
