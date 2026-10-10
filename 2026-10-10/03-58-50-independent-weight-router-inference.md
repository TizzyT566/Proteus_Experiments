# Independent Weight Router: Free-Running Inference

**Experiment completed:** 2026-10-10; final result artifact timestamp 03:58:50 America/New_York (EDT). The filename time is based on the final artifact's filesystem timestamp; the run did not record an exact wall-clock completion time.
**Outcome:** The custom router replaced the native Qwen gate at all 40 MoE layers, with zero native gate forward calls. It completed three free-running generations, but all three responses were repetitive or incomplete and failed the practical language-quality check.

## Objective

Test the trained independent per-expert weight functions in actual autoregressive generation, bypassing Qwen's native router throughout the model. This follows the held-out router evaluation, which had tested fixed hidden states but had not tested a generation loop in which custom routing changes the hidden states subsequently seen by later layers.

## Model and readiness

The model was `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. This is the available public post-trained checkpoint, not the separate Base checkpoint referenced in earlier architecture notes. The model ran without weight quantization in BF16 where supported.

Before loading the model, the runner verified that all 40 layer banks and all 40 calibration entries existed. Each layer bank contained 256 separate expert functions, each a two-layer MLP with a 16-unit ReLU hidden layer and two outputs (predicted rank and predicted weight). The 2,048D gate input was standardized with the per-layer mean and standard deviation saved during training. The learned calibration power was 1.0 in every layer.

The experiment evaluated the predicted weight output only. It applied softplus to each of the 256 predicted weights, selected the largest eight, then normalized those selected values to sum to one. The rank output was ignored. The model's native gate `forward` method was replaced at every MoE layer; an audit counted calls to confirm the native gate was never used to choose experts.

## Prompts and generation procedure

The test used three held-out prompt groups from the existing topic corpus. Inputs were raw prompt text, without a chat template, matching the input format used for the earlier trace/generation benchmark. Prompt lengths were 34, 42, and 41 tokens. Decoding used greedy generation (`do_sample=false`, `num_beams=1`), a maximum of 80 new tokens, and the model's tokenizer EOS and pad IDs. Each response reached the 80-token cap.

### Topic 004 — process and thread

**Prompt:**

> Compare a process and a thread in an operating system. Explain address-space isolation, scheduling, communication cost, and one example where multiple threads are useful inside a single process.

**Custom-router response (80 tokens; 516.31 seconds; 0.15495 tokens/second):**

```text


<think>

Here's a comparison between a **process and a thread** in an operating system, covering the requested aspects:

**Process**
**Thread**
**Address-space isolation**
**Scheduling**
**Communication cost**
**Example**

**Process**
**Thread**
**Address-space isolation**
**Scheduling**
**Communication cost**
**Example**
```

### Topic 005 — water cycle

**Prompt:**

> Summarize the water cycle from evaporation through precipitation and runoff. Include the role of plants and groundwater, and explain why local weather can change even though the overall amount of water on Earth remains roughly constant.

**Custom-router response (80 tokens; 507.90 seconds; 0.15751 tokens/second):**

```text

<think>
Here's a thinking process that leads to the suggested summary:

1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1
```

### Topic 017 — debugging failing tests

**Prompt:**

> A developer changed a shared function and several unrelated tests now fail. Propose a systematic way to inspect the failures, identify the smallest common cause, and decide whether the implementation or the tests need to change.

**Custom-router response (80 tokens; 519.64 seconds; 0.15395 tokens/second):**

```text


<think>

Here is a systematic approach to handle this situation, focusing on the " principle of least common cause" and separating implementation from test logic.

 systematic way to inspect the failures, identify the smallest common cause, and decide whether the implementation or the tests need to change.

### 1. ** systematic way to inspect the failures, identify the smallest common cause, and decide whether the implementation
```

The earlier native-router control for these same prompts produced substantive, task-related text under a 192-token generation cap. That is a qualitative reference only: it was a separate run with a different token cap, not a blinded or otherwise controlled quality comparison. The present custom-router responses are plainly repetitive or cut off before answering the prompts.

## Routing audit

Across all 40 layers, the custom gate received 240 calls per layer and routed 354 token rows per layer. These are different counts: each prompt is processed in a multi-token prefill call, followed by one-token decode calls. The audit recorded **zero native gate forward calls** in every layer. Therefore, the replacement path was active throughout the 40-layer generation runs.

| Layer | Distinct experts selected | Most-selected expert | Selections of that expert |
|---:|---:|---:|---:|
| 0 | 181 | 147 | 100 |
| 1 | 191 | 14 | 72 |
| 2 | 207 | 158 | 222 |
| 3 | 171 | 199 | 157 |
| 4 | 154 | 154 | 176 |
| 5 | 144 | 151 | 178 |
| 6 | 146 | 196 | 175 |
| 7 | 132 | 36 | 163 |
| 8 | 137 | 107 | 210 |
| 9 | 140 | 95 | 299 |
| 10 | 127 | 247 | 166 |
| 11 | 144 | 111 | 169 |
| 12 | 127 | 39 | 136 |
| 13 | 130 | 47 | 253 |
| 14 | 142 | 160 | 181 |
| 15 | 138 | 225 | 125 |
| 16 | 135 | 130 | 250 |
| 17 | 143 | 115 | 211 |
| 18 | 139 | 13 | 175 |
| 19 | 139 | 162 | 172 |
| 20 | 138 | 171 | 176 |
| 21 | 151 | 95 | 278 |
| 22 | 154 | 122 | 156 |
| 23 | 157 | 180 | 167 |
| 24 | 159 | 1 | 263 |
| 25 | 153 | 47 | 269 |
| 26 | 171 | 160 | 228 |
| 27 | 164 | 85 | 150 |
| 28 | 142 | 130 | 287 |
| 29 | 148 | 106 | 163 |
| 30 | 147 | 31 | 182 |
| 31 | 153 | 35 | 201 |
| 32 | 140 | 117 | 153 |
| 33 | 128 | 127 | 234 |
| 34 | 121 | 182 | 111 |
| 35 | 150 | 167 | 162 |
| 36 | 158 | 74 | 116 |
| 37 | 149 | 55 | 136 |
| 38 | 130 | 209 | 314 |
| 39 | 143 | 70 | 96 |

Each layer made 240 calls and routed 354 rows; those values are constant across this table and are stated once above. The number of distinct selected experts ranged from 121 to 207, averaging 148.1 out of 256. The most-selected expert in a layer was selected between 72 and 314 times across 354 rows; the maximum is 88.7% of rows. This concentration is a useful warning sign, though this small audit alone does not establish that it caused the bad text. The selected mixture weights summed to one within a maximum absolute error of `2.3842e-7` across all layers.

## Runtime and device placement

| Measurement | Result |
|---|---:|
| Model load | 450.39 s (7.51 min) |
| Generation, all three prompts | 1,543.87 s (25.73 min) |
| Generated tokens | 240 |
| Aggregate generation throughput | 0.15546 tokens/s |
| Total measured load plus generation | 1,994.26 s (33.24 min) |

The model's automatic device map placed layers 0–15 on `cuda:0`, layers 16–27 on `cuda:1`, and layers 28–39 on CPU; the remaining listed text-model components were also CPU-mapped. Router MLPs for CPU-mapped layers were evaluated on `cuda:0`. This experiment did not measure router-only latency separately or compare routing latency with the native gate. The low end-to-end throughput includes the model's CPU layer placement, so it cannot be attributed to the independent router alone.

## Interpretation and limitations

The engineering objective succeeded: all trained banks were present, the model ran, all 40 native gate forwards were bypassed, top-eight IDs and normalized weights were produced, and three autoregressive generations completed. The quality objective failed: none of the responses gave a useful answer.

A plausible explanation is distribution shift compounded across depth and time. The functions were trained against native-router targets on captured prompt-prefill states. In free-running generation, replacing the gate changes expert outputs and therefore changes hidden states passed to later layers; during decoding, generated tokens also create new states. The functions may consequently receive inputs unlike their training examples, with each layer's changed choice influencing subsequent routing. This run is consistent with that explanation but does not isolate it from other issues, such as imbalanced predicted weights or the difference between imitating the native gate's normalized weights and measuring an expert's actual contribution to output quality.

This was a three-prompt generation check on one post-trained checkpoint. It did not measure perplexity, task accuracy with a rubric, repeated-seed variation, causal per-layer effects, or a controlled native-versus-custom quality score. It also used no weight quantization, so it isolates the custom routing path from the separate INT4 experiment. The completion of generations and normalization of selected weights establish that the integration executed; they do not establish that the learned scores are reliable outside the fixed-state training/evaluation distribution.
