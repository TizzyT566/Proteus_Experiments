# Independent per-expert rank and weight router

## Objective

Test whether every expert can have its own learned function from the full 2,048D hidden state to two outputs: its predicted rank and its predicted mixture weight. During held-out routing, evaluate all 256 functions for the layer, choose eight experts from those predictions, and leave the original gate out of the selection path.

## Model, traces, and split

The reference model was `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. The experiment used the already captured full-precision router traces: 22,296 prompt-token states per layer, with 2,048D hidden inputs, all 256 reference scores, the native selected IDs, and their normalized weights. Those traces contain 48 underlying topic groups. The outer split held out 10 groups and trained on 38, using seed 23; this produced 4,504 held-out tokens per layer, or 180,160 layer-token decisions across 40 layers. A further 90/10 group split inside the training groups was used for checkpoint selection and weight calibration.

These are raw prompt-prefill states from the existing corpus. They are not chat-template, generated-token, cached-decode, or free-running states.

## Method

Each of the 40 layers had 256 separate expert functions. One function was a two-layer MLP with a 16-unit ReLU hidden layer and two outputs. It received the layer-normalized 2,048D hidden state and had its own parameters for that expert. The implementation evaluated the 256 functions together as a batched GPU operation; it did not share their learned parameters.

The rank target was the expert’s ordinal position among the native gate’s 256 scores, scaled to `[0,1]`. The captured native top-eight tuple supplied the first eight positions to preserve the gate’s actual tie choices. The weight target was the captured native mixture weight for a selected expert and zero for every unselected expert.

Training used rank smooth-L1 loss and a weight loss on positive and negative targets, AdamW at learning rate `0.001`, weight decay `0.0001`, batches of 2,048, 24 epochs, and gradient clipping at 1. Each layer’s checkpoint was selected on internal validation by top-eight recall, exact-set rate, then weight error. A single per-layer power calibrated the predicted positive weights on validation; after choosing the top eight, those predicted weights were normalized to sum to one.

For the main test, the rank output selected the eight experts and the weight output assigned their mixture weights. A second diagnostic selected experts using the weight output itself. Both paths evaluated all 256 functions and did not call the native gate for selection. The native outputs were used only to form training targets and calculate held-out metrics.

The fitted bank contains 10,240 functions, or 8,401,408 parameters per layer and 336,056,320 across the 40 layers. The 40 saved layer checkpoints occupy 1,345,002,280 bytes.

## Held-out results

Recall is the mean fraction of the native top eight present in the predicted top eight. Exact-set rate is the fraction of tokens where all eight IDs match. Full-dense weight L1 sums absolute weight error across all 256 experts, including missing native weight mass when a selected expert is missed. The affine baseline is the earlier weight-only affine predictor evaluated exhaustively over all 256 candidates on the same outer held-out split.

| Selector and weight output | Top-8 recall | Exact-set rate | Full-dense weight L1 | Full-dense weight RMSE |
|---|---:|---:|---:|---:|
| Rank output selects; learned weight output assigns weights | 0.6780 | 0.0905 | 0.6136 | 0.01583 |
| Weight output selects and assigns weights | 0.7550 | 0.1403 | 0.4690 | 0.01294 |
| Affine weight-only baseline | 0.7878 | 0.3482 | 0.4670 | 0.01301 |

The rank head’s average Spearman correlation with all native scores was 0.952, while its top-eight recall was 0.678. It tracked the broad ordering well but did not reliably separate the experts around the top-eight cutoff. Its native-selected top-eight exact-set rate was 9.0%.

Using the predicted weight output as the selection score improved recall by 7.7 percentage points over the rank output and reduced full-dense weight L1 from 0.6136 to 0.4690. Against the affine baseline, however, it lost 3.3 points of recall and 20.8 points of exact-set rate; its weight L1 was higher by 0.0020. It had lower weight L1 than the affine baseline in 20 of 40 layers, and did not beat it on recall or exact-set rate in any layer.

The complete per-layer measurements follow. Each row is the held-out result for that layer; weight-output selection uses the learned weight head both to rank and weight its top eight.

| Layer | Rank recall | Rank exact | Rank L1 | Weight recall | Weight exact | Weight L1 | Affine recall | Affine exact | Affine L1 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.722 | 0.125 | 0.489 | 0.809 | 0.136 | 0.330 | 0.853 | 0.471 | 0.371 |
| 1 | 0.727 | 0.128 | 0.504 | 0.782 | 0.137 | 0.404 | 0.819 | 0.417 | 0.426 |
| 2 | 0.712 | 0.091 | 0.546 | 0.724 | 0.061 | 0.498 | 0.763 | 0.380 | 0.549 |
| 3 | 0.694 | 0.079 | 0.591 | 0.742 | 0.114 | 0.489 | 0.753 | 0.313 | 0.569 |
| 4 | 0.692 | 0.111 | 0.549 | 0.764 | 0.087 | 0.409 | 0.822 | 0.408 | 0.407 |
| 5 | 0.681 | 0.116 | 0.595 | 0.740 | 0.104 | 0.475 | 0.776 | 0.354 | 0.489 |
| 6 | 0.674 | 0.087 | 0.613 | 0.757 | 0.106 | 0.472 | 0.786 | 0.373 | 0.481 |
| 7 | 0.669 | 0.064 | 0.635 | 0.753 | 0.142 | 0.495 | 0.772 | 0.345 | 0.515 |
| 8 | 0.664 | 0.075 | 0.642 | 0.746 | 0.117 | 0.503 | 0.757 | 0.340 | 0.527 |
| 9 | 0.679 | 0.098 | 0.627 | 0.758 | 0.114 | 0.478 | 0.779 | 0.343 | 0.487 |
| 10 | 0.673 | 0.091 | 0.610 | 0.746 | 0.121 | 0.477 | 0.805 | 0.361 | 0.435 |
| 11 | 0.680 | 0.116 | 0.604 | 0.750 | 0.087 | 0.470 | 0.812 | 0.367 | 0.427 |
| 12 | 0.674 | 0.093 | 0.632 | 0.732 | 0.116 | 0.514 | 0.766 | 0.356 | 0.522 |
| 13 | 0.689 | 0.073 | 0.600 | 0.754 | 0.134 | 0.482 | 0.772 | 0.335 | 0.500 |
| 14 | 0.681 | 0.062 | 0.613 | 0.730 | 0.154 | 0.530 | 0.744 | 0.338 | 0.572 |
| 15 | 0.672 | 0.083 | 0.633 | 0.739 | 0.138 | 0.520 | 0.762 | 0.315 | 0.535 |
| 16 | 0.671 | 0.068 | 0.651 | 0.740 | 0.122 | 0.499 | 0.772 | 0.342 | 0.513 |
| 17 | 0.656 | 0.046 | 0.668 | 0.728 | 0.144 | 0.533 | 0.767 | 0.355 | 0.520 |
| 18 | 0.668 | 0.079 | 0.651 | 0.723 | 0.123 | 0.539 | 0.757 | 0.322 | 0.527 |
| 19 | 0.664 | 0.080 | 0.658 | 0.720 | 0.159 | 0.551 | 0.730 | 0.315 | 0.575 |
| 20 | 0.654 | 0.072 | 0.677 | 0.720 | 0.140 | 0.559 | 0.739 | 0.307 | 0.575 |
| 21 | 0.663 | 0.083 | 0.661 | 0.721 | 0.074 | 0.523 | 0.768 | 0.319 | 0.522 |
| 22 | 0.677 | 0.093 | 0.614 | 0.766 | 0.151 | 0.456 | 0.806 | 0.321 | 0.461 |
| 23 | 0.689 | 0.110 | 0.601 | 0.754 | 0.139 | 0.466 | 0.786 | 0.323 | 0.457 |
| 24 | 0.663 | 0.067 | 0.646 | 0.731 | 0.115 | 0.499 | 0.765 | 0.328 | 0.522 |
| 25 | 0.679 | 0.095 | 0.609 | 0.749 | 0.131 | 0.460 | 0.787 | 0.322 | 0.450 |
| 26 | 0.663 | 0.059 | 0.646 | 0.736 | 0.137 | 0.509 | 0.756 | 0.331 | 0.521 |
| 27 | 0.665 | 0.075 | 0.650 | 0.736 | 0.130 | 0.515 | 0.761 | 0.299 | 0.521 |
| 28 | 0.683 | 0.092 | 0.604 | 0.753 | 0.137 | 0.437 | 0.791 | 0.326 | 0.425 |
| 29 | 0.670 | 0.071 | 0.614 | 0.755 | 0.160 | 0.464 | 0.783 | 0.340 | 0.457 |
| 30 | 0.685 | 0.102 | 0.605 | 0.747 | 0.160 | 0.479 | 0.789 | 0.314 | 0.443 |
| 31 | 0.671 | 0.053 | 0.634 | 0.773 | 0.183 | 0.453 | 0.787 | 0.327 | 0.447 |
| 32 | 0.692 | 0.118 | 0.577 | 0.787 | 0.173 | 0.413 | 0.835 | 0.356 | 0.351 |
| 33 | 0.675 | 0.102 | 0.600 | 0.778 | 0.114 | 0.404 | 0.839 | 0.384 | 0.352 |
| 34 | 0.696 | 0.120 | 0.566 | 0.813 | 0.200 | 0.355 | 0.860 | 0.399 | 0.302 |
| 35 | 0.665 | 0.108 | 0.633 | 0.760 | 0.194 | 0.475 | 0.775 | 0.310 | 0.469 |
| 36 | 0.665 | 0.123 | 0.651 | 0.778 | 0.204 | 0.453 | 0.802 | 0.331 | 0.420 |
| 37 | 0.660 | 0.090 | 0.658 | 0.779 | 0.208 | 0.450 | 0.815 | 0.349 | 0.398 |
| 38 | 0.672 | 0.121 | 0.618 | 0.790 | 0.194 | 0.402 | 0.838 | 0.371 | 0.354 |
| 39 | 0.692 | 0.099 | 0.572 | 0.832 | 0.255 | 0.322 | 0.863 | 0.417 | 0.284 |

## Router-only timing

On an NVIDIA GeForce RTX 5090, the benchmark evaluated one token at a time through all 40 layers, synchronized after each decision, and measured two repetitions over 32 held-out contexts per layer. The native path used the native 256-by-2,048 gate and top-eight normalization. The independent path evaluated every expert MLP, selected by its rank output, then used its weight output. Expert MLP execution and the rest of the model were excluded.

| Router | Median time per token across 40 layers | Median time per layer decision |
|---|---:|---:|
| Native gate | 21.99 ms | 549.8 μs |
| Independent expert MLPs | 85.42 ms | 2,135.6 μs |

Evaluating all 256 MLP functions took 3.88 times as long as the native gate in this one-token benchmark, adding 63.43 ms per token across the 40 routing decisions.

## Interpretation and limits

This experiment shows that per-expert functions can be trained and evaluated without executing the expert weights or calling the native router at test time. The rank output, as trained here, was a weak top-eight selector. The weight output was a much stronger selector and nearly matched the affine predictor’s full-dense mixture-weight error, but it still selected fewer of the native experts and matched the exact set much less often.

The training targets describe the existing fixed 256-expert router. A rank is defined relative to those 256 experts, and each native mixture weight is normalized among the native selected set. Therefore these scores do not yet establish an absolute, registry-independent usefulness scale for packages trained separately or for an arbitrary changing collection. The experiment tests whether independent functions can reproduce this native router, not whether they compose correctly as new experts are added.

The held-out test reused hidden states produced by the native model. It did not run a free-generation pass with this router replacing Qwen’s gate at all 40 layers. Such a run would feed altered expert outputs into later layers and create states outside this fixed-trace evaluation. No generation-quality result is claimed here.

The training run started at 02:25:03 and finished at 02:47:01 EDT on 2026-10-10, taking 1,317.4 seconds. A separate held-out re-evaluation and router-only benchmarking pass took 168.2 seconds; its exact wall-clock timestamps were not recorded.

