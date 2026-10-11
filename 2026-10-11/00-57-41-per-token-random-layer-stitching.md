# Per-Token Random Qwen–Ornith Layer Stitching

**Run window:** 2026-10-10 23:45:52 to 2026-10-11 00:57:41 America/New_York (EDT). The completion time is converted from the recorded UTC timestamp `2026-10-11T04:57:41Z`.

**Result:** Independent random layer choices on every token produced four on-topic, readable openings, each capped at 128 output tokens. The stitched generations differed from Qwen on all four prompts and averaged 1,017.21 seconds per prompt, 54.2% longer than the Qwen baseline. This is evidence that this one severe mixture could execute and produce locally coherent text; it is not a broad compatibility or quality result.

## Objective

Test the clarified token-wise stitching condition: for every input token, independently choose Qwen or Ornith for each of the 40 decoder layers, then make a fresh choice for every layer on the next token. This applies to prompt processing and autoregressive generation. The test asks whether a model whose layer donors change within a token sequence can still produce readable, relevant continuations.

This is a full decoder-layer experiment. Each selected layer brings its normalization, attention or linear-attention block, MoE gate/router, routed expert weights, and shared-expert weights. The native router belonging to that selected layer remains in use. It is not a test of an external router, an arbitrary expert registry, or swapping only an eight-expert bundle.

## Checkpoints and model

| Role | Checkpoint | Revision | Precision |
|---|---|---|---|
| Active model and reference | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | BF16 |
| Per-layer donor | `ornith-ai/Ornith-1.5-35B-A3B` | `10fbf86fed7ecee4a061f8b499a618f46001cac1` | BF16 |

The available Qwen checkpoint is post-trained, not a Base checkpoint. Both checkpoints expose the same 40-layer Qwen3.5-MoE decoder structure and matching names/shapes for the layer tensors used by the run. The model has 2,048 hidden dimensions and 256 routed experts per layer with top-eight native expert routing. Matching shapes establish loadability, not equivalent hidden-state meanings.

Only one Transformers model instance was loaded: Qwen. Ornith layer tensors were read from local safetensors and supplied to that active model as choices were applied; both complete models were never resident together. The Qwen token embeddings, final normalization, and output head remained in place. Every decoder layer, including its native router and experts, could independently come from either checkpoint.

## Data and protocol

The evaluation reused the established four-item broad-topic suite and the saved Qwen-tokenized input IDs. There was no training split, sampling of prompts, repeated random seed, or formal human grading. Each prompt generated exactly 128 tokens and reached the configured output cap.

| Prompt ID | Topic | Prompt tokens | Qwen output tokens | Stitched output tokens |
|---|---|---:|---:|---:|
| `topic_000` | Autumn leaf pigments | 65 | 128 | 128 |
| `topic_010` | REST API inventory search | 55 | 128 | 128 |
| `topic_025` | Bus-lane evaluation | 58 | 128 | 128 |
| `topic_045` | Symmetric and asymmetric encryption | 42 | 128 | 128 |

Decoding was BF16, greedy (`argmax`, equivalent to `do_sample=false`, one beam), with cache enabled. The same saved prompt token IDs were fed one at a time. Before each model forward, the runner drew a new 40-bit mask using Python `random.Random` with experiment seed `20261010` XOR the first eight SHA-256 bytes of the prompt ID, interpreted little-endian. Each bit is an independent Bernoulli(0.5): `0` chooses Qwen and `1` chooses Ornith. The complete masks were recorded for reproducibility.

The 728 forwards comprise 220 prompt-token forwards plus 127 forwards for each of four 128-token continuations. The first generated token is predicted from the final prompt forward, so the final generated token needs no additional forward. At 40 independent donor decisions per forward, this produced 29,120 layer-token assignments. Ornith was selected 14,601 times (50.14%); Qwen was selected 14,519 times. The recorded donor changes from the initial Qwen state across prompts were 7,262 transitions to Qwen and 7,350 to Ornith.

The donor changes at every layer were applied before that layer ran. Attention KV state and the linear-attention recurrent state persisted across token steps and therefore across donor changes. The whole prompt was processed token by token so that it obeyed the same per-token rule as generated text.

## Runtime environment

The experiment ran on Windows with BF16 CUDA inference. Automatic placement used CUDA device 0 (RTX 5090) for decoder layers 0–15 and CUDA device 1 (RTX 3090) for layers 16–27. Layers 28–39 were CPU-offloaded in the device map and materialized for execution through Accelerate; embeddings were on device 0, while the final norm, rotary embedding module, and output head were mapped to CPU. The configured memory ceilings were 28 GiB on device 0, 20 GiB on device 1, and 400 GiB for CPU placement. TF32 was disabled.

Optimized `causal_conv1d` and `flash-linear-attention` kernels were unavailable, so the model used reference PyTorch implementations. This and CPU offload affect absolute runtime. Dynamic switching also requires streaming or swapping complete layer weights; the timing is not a pure router lookup benchmark.

## Generation-time comparison

Per-prompt elapsed time includes prompt processing and output generation, but excludes the one-time model load. The dynamic model load took 88.08 seconds. The Qwen reference recorded an 85.98-second load for its resumed three-prompt pass; the previously generated first Qwen response was reused. Load time is excluded from the following generation comparison.

| Prompt | Qwen seconds | Random per-token stitch seconds | Difference | Slower elapsed time | Exact token sequence match? |
|---|---:|---:|---:|---:|:---:|
| Autumn leaf pigments | 671.91 | 1210.43 | +538.52 s | +80.1% | No |
| REST API inventory search | 662.62 | 977.99 | +315.37 s | +47.6% | No |
| Bus-lane evaluation | 650.72 | 981.60 | +330.88 s | +50.8% | No |
| Symmetric and asymmetric encryption | 653.76 | 898.82 | +245.05 s | +37.5% | No |

| Aggregate metric (four prompts) | Qwen BF16 reference | Random per-token stitch |
|---|---:|---:|
| Generated tokens | 512 | 512 |
| Total generation time | 2639.02 s (43.98 min) | 4068.84 s (67.81 min) |
| Mean generation time per prompt | 659.76 s | 1017.21 s |
| Aggregate output rate (tokens / total generation seconds) | 0.1940 tokens/s | 0.1258 tokens/s |

The random per-token condition took 357.45 more seconds per prompt on average (54.2% longer elapsed time). Its aggregate generation rate was 35.1% lower. These figures combine the costs of per-token prompt processing, layer donor changes, offload, and reference kernels; they do not isolate a router’s decision latency.

All four output token sequences differed from the Qwen reference (0/4 exact matches). This measures sequence divergence, not which answer is better.

## Per-layer donor frequency

Each layer received one donor bit per forward, so its denominator is 728. With independent fair choices, the expected Ornith count per layer is 364. The observed range was 329–396 (45.2%–54.4%), consistent with sampling variation in this one schedule.

| Layer | Ornith assignments | Ornith share | Qwen assignments |
|---:|---:|---:|---:|
| 0 | 356 | 48.9% | 372 |
| 1 | 388 | 53.3% | 340 |
| 2 | 366 | 50.3% | 362 |
| 3 | 347 | 47.7% | 381 |
| 4 | 357 | 49.0% | 371 |
| 5 | 386 | 53.0% | 342 |
| 6 | 367 | 50.4% | 361 |
| 7 | 348 | 47.8% | 380 |
| 8 | 355 | 48.8% | 373 |
| 9 | 359 | 49.3% | 369 |
| 10 | 369 | 50.7% | 359 |
| 11 | 352 | 48.4% | 376 |
| 12 | 349 | 47.9% | 379 |
| 13 | 346 | 47.5% | 382 |
| 14 | 329 | 45.2% | 399 |
| 15 | 355 | 48.8% | 373 |
| 16 | 348 | 47.8% | 380 |
| 17 | 370 | 50.8% | 358 |
| 18 | 377 | 51.8% | 351 |
| 19 | 371 | 51.0% | 357 |
| 20 | 357 | 49.0% | 371 |
| 21 | 364 | 50.0% | 364 |
| 22 | 348 | 47.8% | 380 |
| 23 | 379 | 52.1% | 349 |
| 24 | 387 | 53.2% | 341 |
| 25 | 358 | 49.2% | 370 |
| 26 | 350 | 48.1% | 378 |
| 27 | 373 | 51.2% | 355 |
| 28 | 352 | 48.4% | 376 |
| 29 | 387 | 53.2% | 341 |
| 30 | 393 | 54.0% | 335 |
| 31 | 359 | 49.3% | 369 |
| 32 | 396 | 54.4% | 332 |
| 33 | 368 | 50.5% | 360 |
| 34 | 376 | 51.6% | 352 |
| 35 | 358 | 49.2% | 370 |
| 36 | 385 | 52.9% | 343 |
| 37 | 376 | 51.6% | 352 |
| 38 | 371 | 51.0% | 357 |
| 39 | 369 | 50.7% | 359 |

## Output observations

The four stitched responses began with relevant, readable material: an explanation of leaf pigments, a product-search API schema, a bus-lane evaluation plan, and an encryption comparison. Each decoded continuation stopped at 128 tokens, often mid-section or mid-sentence. No output was scored with a factuality, completeness, coherence, or preference rubric; the observations are qualitative. The decoded Qwen and stitched continuations are included below so the comparison can be read without other artifacts. Trailing whitespace is trimmed for Markdown rendering; token IDs and equality checks use the original generated sequences.

## Interpretation and limitations

- The run shows that this particular random, layer-by-layer and token-by-token mixture loaded and generated four relevant short continuations. It offers limited evidence that some cross-checkpoint layer transitions can remain locally intelligible under a severe perturbation.
- This does not show that all layer combinations, arbitrary fine-tunes, or external expert packages are interchangeable. The test uses two related 35B-A3B checkpoints with matching architecture and shapes, one random schedule, and four prompts.
- Each donor choice replaces a complete decoder block, including attention, normalization, router/gate, and experts. The experiment cannot identify which of those components supports or harms compatibility. It is not an experiment on swapping only experts while holding a shared backbone and router fixed.
- Cache state is carried from one token to the next while donors change. This is part of the tested token-wise setting, but it means the experiment does not test a fresh cache for each donor or guarantee that their recurrent/KV representations are mutually aligned.
- Every continuation hit the 128-token cap. The results cannot establish complete-answer quality, long-horizon stability, or behavior after hundreds or thousands of tokens.
- The run measures end-to-end inference time, not routing-only latency. It includes one-token prompt processing, layer-weight switching, CPU offload, and fallback kernels; it cannot be read as a direct comparison of a redesigned router’s compute cost with Qwen’s native router.

## Exact prompts and generated continuations

### `topic_000` — Autumn leaf pigments

**Prompt (65 tokens):** Explain why leaves change color in autumn. Describe the roles of chlorophyll, carotenoids, and anthocyanins, then explain how temperature and daylight affect the timing. Use plain language and distinguish what is well established from what varies by tree species.

**Qwen BF16 reference — 128 tokens; 671.91 seconds:**

~~~~text
Here is an explanation of why leaves change color in autumn, broken down by the science behind the pigments and the environmental triggers.

### 1. The Core Mechanism: Why Leaves Change Color
Leaves change color not because new colors are "created" in autumn, but because the **green pigment disappears**, revealing colors that were already there or newly produced.

In spring and summer, leaves are green because they are busy making food (sugar) through **photosynthesis**. This process requires a green pigment called **chlorophyll**. Chlorophyll is dominant and masks other pigments present in the leaf.

As autumn
~~~~

**Random per-token stitch — 128 tokens; 1210.43 seconds:**

~~~~text
Here is an explanation of why leaves change color in autumn, broken down by the science involved, the pigments at play, and the environmental triggers.

### The Short Answer
Leaves change color because **chlorophyll breaks down**, revealing other colors that were already present, while new colors are sometimes produced. This process is triggered by shorter days and cooler temperatures, which signal the tree to prepare for winter.

---

### 1. The Pigments: What Colors Are We Seeing?

Leaves contain several types of pigments. The color you see depends on which pigments are dominant at a given time.

####
~~~~

### `topic_010` — REST API inventory search

**Prompt (55 tokens):** Design a REST API endpoint for searching product inventory by category, minimum stock, and a free-text query. Specify the HTTP method, path, query parameters, example response, validation behavior, and how pagination should work.

**Qwen BF16 reference — 128 tokens; 662.62 seconds:**

~~~~text
# REST API Endpoint: Search Product Inventory

## Endpoint Specification

### HTTP Method & Path

```
GET /api/v1/products/search
```

### Query Parameters

| Parameter       | Type    | Required | Description                                      | Validation Rules                          |
|-----------------|---------|----------|--------------------------------------------------|-------------------------------------------|
| `category`      | string  | No       | Filter by product category                       | Alphanumeric, hyphens, underscores; max 100 chars |
| `min_stock`     | integer | No       | Minimum stock level to filter by                 | ≥ 0; max
~~~~

**Random per-token stitch — 128 tokens; 977.99 seconds:**

~~~~text
# REST API Endpoint: Search Product Inventory

## Endpoint Specification

### HTTP Method & Path

```
GET /api/v1/products/search
```

### Query Parameters

| Parameter | Type | Required | Description | Validation Rules |
|-----------|------|----------|-------------|------------------|
| `category` | string | Optional | Product category filter | Alphanumeric, hyphens, underscores; max 100 chars |
| `min_stock` | integer | Optional | Minimum stock level | Integer ≥ 0; max 1,000,000 |
| `q` |
~~~~

### `topic_025` — Bus-lane evaluation

**Prompt (58 tokens):** A city is considering a bus lane on a congested street. Outline an evaluation plan using travel time, reliability, passenger counts, safety, and effects on nearby streets. Explain how to compare results fairly before and after the change.

**Qwen BF16 reference — 128 tokens; 650.72 seconds:**

~~~~text
Here is a comprehensive evaluation plan for assessing the impact of a new bus lane on a congested street. This plan focuses on quantitative data collection, comparative analysis, and fair benchmarking.

### **1. Evaluation Plan Overview**

The evaluation will be conducted in three phases: **Baseline (Pre-Implementation)**, **Implementation (Short-term, 1–3 months)**, and **Stabilization (Long-term, 6–12 months)**.

#### **A. Key Metrics & Data Collection Methods**

| Metric | Definition | Data Collection Method |
| :--- | :--- | :--- |
| **1
~~~~

**Random per-token stitch — 128 tokens; 981.60 seconds:**

~~~~text
Here is a comprehensive evaluation plan for assessing the impact of a new bus lane on a congested street. This plan utilizes a **Before-After Control-Impact (BACI)** study design, which is the gold standard for transportation impact assessments to ensure fair and statistically valid comparisons.

---

### **Evaluation Plan: Bus Lane Impact Assessment**

#### **1. Study Design: Before-After Control-Impact (BACI)**
To compare results fairly, we must isolate the effect of the bus lane from other variables (e.g., seasonal traffic changes, economic shifts, or concurrent construction).

*   **
~~~~

### `topic_045` — Symmetric and asymmetric encryption

**Prompt (42 tokens):** Explain the difference between symmetric and asymmetric encryption, and why real secure protocols often use both. Include the role of a session key and authenticated encryption.

**Qwen BF16 reference — 128 tokens; 653.76 seconds:**

~~~~text
To understand modern cryptography, one must distinguish between the two primary methods of encryption and understand how they are combined to create secure communication channels (such as HTTPS/TLS).

Here is the step-by-step logical deduction of the differences, the necessity of combining them, and the specific roles of session keys and authenticated encryption.

### 1. Symmetric vs. Asymmetric Encryption

The fundamental difference lies in the number of keys used and the mathematical operations involved.

#### Symmetric Encryption
*   **Mechanism:** Uses a **single shared key** for both encryption and decryption.
*   **Analogy:** A physical padlock
~~~~

**Random per-token stitch — 128 tokens; 898.82 seconds:**

~~~~text
Here is a detailed explanation of the differences between symmetric and asymmetric encryption, the rationale for combining them, and the specific roles of session keys and authenticated encryption.

---

### 1. Symmetric Encryption

**Definition:**
Symmetric encryption (also known as **secret-key encryption**) uses a **single key** for both encrypting and decrypting data. The sender and the receiver must share this same secret key beforehand.

**Characteristics:**
*   **Speed:** Extremely fast computationally. It is efficient for encrypting large amounts of data (bulk data).
*   **Key Management:** The primary challenge is **
~~~~
