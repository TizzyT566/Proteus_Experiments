# Independent expert rank/weight router smoke test

## Objective and scope

This pilot checked whether the independent per-expert rank/weight training and evaluation path could run end to end. It trained and evaluated layer 0 only, for one epoch. For each token, 256 expert-specific functions predicted a rank and a mixture weight from the same 2,048D hidden state; the predicted rank selected eight experts, and the predicted weights assigned their mixture weights. The native router was not called to select experts at test time. Its captured outputs supplied offline training labels and evaluation targets.

This was an implementation smoke test, not a complete 40-layer evaluation or a free-running language-generation test.

## Model, data, and split

The reference checkpoint was `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. The inputs were captured raw prompt-prefill hidden states, not generated-token or cached-decode states. Layer 0 had 22,296 token states from 48 topic groups, each with the 2,048D hidden state and the native scores, selected expert IDs, and normalized top-eight weights.

The group split used seed 23. It held out 10 groups (4,504 tokens) and used 38 groups (17,792 tokens) for training and validation. Within those 38 groups, 34 groups (15,960 tokens) fit the model and 4 groups (1,832 tokens) selected the validation calibration. Held-out groups were not used for training or calibration. Since only one layer and one split were evaluated, these are token-averaged point estimates, not estimates of between-layer or between-seed variation.

## Router and training procedure

Each of the 256 experts had a separate MLP with architecture 2,048 → 16 → 2 and a ReLU hidden layer. One output predicted the expert's native-score ordinal rank scaled to `[0,1]`; the other predicted its native mixture weight when selected and zero otherwise. Each function had 32,818 parameters, for 8,401,408 parameters in the layer's 256-function bank.

Training ran for one epoch with AdamW, learning rate `0.001`, weight decay `0.0001`, batch size 4,096, and gradient clipping at 1. The rank loss was smooth L1. The weight loss was squared error on positive targets plus `0.1` times squared error on zero targets; the two losses were added with weight-loss scale 1. The weight output passed through softplus. No clustering or PCA was used.

At evaluation, all 256 functions were evaluated for each held-out state. The eight largest rank outputs selected the experts. Their softplus weight outputs were restricted to those eight and normalized to sum to one. Validation selected a power of 4 for the weight outputs before normalization; applying this power changes weights but not the rank-based selected IDs.

## Held-out results

Recall is the mean fraction of the native top eight present in the predicted top eight. Exact-set rate is the fraction of tokens for which all eight IDs match. Full-dense weight L1 is the per-token sum of absolute error across all 256 experts, averaged over tokens; predicted weights are zero for unselected experts. Full-dense weight RMSE is the square root of mean squared error across tokens and all 256 experts. Rank Spearman is the per-token Spearman correlation across all 256 expert scores, averaged over tokens. Native-selected rank Spearman computes that correlation only on the eight native-selected experts.

| Split and weight treatment | Top-8 recall | Exact-set rate | Full-dense weight L1 | Full-dense weight RMSE | Rank Spearman, all experts | Rank Spearman, native top 8 |
|---|---:|---:|---:|---:|---:|---:|
| Validation, raw predicted weights | 0.0895 | 0.0000 | 1.8623 | 0.03203 | 0.7445 | -0.1657 |
| Validation, power-4 calibrated weights | 0.0895 | 0.0000 | 1.8587 | 0.03212 | 0.7445 | -0.1657 |
| Held out, raw predicted weights | 0.0958 | 0.0000 | 1.8531 | 0.03186 | 0.7530 | -0.1486 |
| Held out, power-4 calibrated weights | 0.0958 | 0.0000 | 1.8493 | 0.03194 | 0.7530 | -0.1486 |

The power-4 calibration left selection metrics unchanged, as expected for rank-based selection. On held-out tokens it reduced L1 by 0.00376 (about 0.20%) while increasing RMSE by 0.000081. The calibrated held-out recall corresponds to about 0.77 of the eight native-selected experts recovered per token on average, and no held-out token had an exact top-eight match.

## Runtime and interpretation

The run completed in 22.14 seconds on an NVIDIA GeForce RTX 5090 (`cuda:0`). It trained only layer 0 for one epoch. The saved layer checkpoint was 33,625,057 bytes. This timing does not measure full-model inference, generation quality, or routing latency against the native router.

The all-expert rank scores had a positive mean Spearman correlation of 0.753 with the native ordering, but the selector performed poorly at the top-eight cutoff: recall was 0.0958 and exact-set rate was zero. Correlation over all 256 candidates therefore did not imply useful top-eight choices in this pilot. The negative correlation restricted to the native top eight further shows that the rank head did not preserve the ordering where selection matters most. Weight calibration barely changed dense error and could not repair missed expert IDs.

The one-epoch setting was chosen for a smoke test, so these results do not establish the performance ceiling of the approach or isolate the effect of training duration. They do establish that this pilot configuration was not a useful replacement router. The experiment tested only layer 0, one data split, and prefill states; it did not test other layers, repeated seeds, generated-token states, end-to-end text quality, or performance with an open-ended expert registry.

## Run chronology

The run started at 2026-10-10 02:23:58.965 and finished at 2026-10-10 02:24:21 local time (UTC−04:00). The report filename uses the recorded local completion time.
