# Four-Prompt Qwen–Ornith Layer-Swap Comparison

**Experiment completed:** 2026-10-10 08:00:04 America/New_York (EDT). The final comparison artifact was recorded at 08:00:05 EDT.

**Outcome:** The hybrid loaded and completed all four generations. Its generated token continuation differed from the Qwen reference on all four prompts. Mean generation time was 787.88 seconds per prompt for the hybrid and 659.76 seconds for Qwen, making the hybrid 19.4% slower in this run. Every response reached the 128-token cap, so the experiment does not establish which model answered better.

## Objective

Test whether contiguous middle layers from the Ornith-1.5 checkpoint can be substituted into Qwen3.6 without additional training, then compare free-running outputs with the Qwen checkpoint on the same four broad-topic prompts. The prompt set was reduced to four prompts per model before completing the comparison.

The practical question was whether the hybrid could load and generate, and how its short responses and generation speed compared with the Qwen reference. This was a direct layer-swap check, not a fine-tuning experiment or a controlled test of an interchangeable expert registry.

## Checkpoints and hybrid construction

| Role | Checkpoint | Revision | Precision |
|---|---|---|---|
| Reference and hybrid backbone | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | BF16 |
| Source for substituted layers | `ornith-ai/Ornith-1.5-35B-A3B` | `10fbf86fed7ecee4a061f8b499a618f46001cac1` | BF16 |

The available Qwen3.6 checkpoint is **post-trained**, not a Base checkpoint. Both checkpoints had matching tensor names and shapes for the layers used; the shared architecture has 40 decoder layers, a 2,048-dimensional hidden state, and 256 routed experts with top-eight routing. The tokenizer vocabulary size was 248,077. A shape preflight passed for all 69 tensors selected for replacement.

The hybrid was assembled offline by replacing Qwen decoder layers **16, 17, 18, and 19** with the corresponding complete Ornith layers. The replacement includes attention, input and post-attention normalization, the MoE router/gate, routed experts, and shared experts. Of the 69 replaced tensors, four were router/gate tensors and eight were expert-weight tensors. Four of Qwen's 26 safetensor shards were modified; 35 unchanged shard files were hard-linked into the hybrid directory.

This is a contiguous four-layer swap without adaptation. It changes more than the experts: it also changes the attention, normalization, and routing computation in those blocks. Matching tensor shapes and vocabulary make the model loadable, but do not guarantee that Ornith layers interpret Qwen activations compatibly or that the surrounding Qwen layers can use Ornith outputs effectively.

## Prompts and protocol

The four prompts were selected from a broader topic corpus to cover biology, API design, transit policy, and security. Each was formatted with the Qwen tokenizer's chat template as a user message, with a generation prompt and thinking disabled when supported. The exact saved prompt token IDs were reused for both conditions.

| Topic ID | Topic | Input tokens |
|---|---|---:|
| `topic_000` | Autumn leaf pigments | 65 |
| `topic_010` | REST API inventory search | 55 |
| `topic_025` | Bus-lane evaluation | 58 |
| `topic_045` | Symmetric and asymmetric encryption | 42 |

Both conditions used BF16 weights, cached autoregressive decoding, greedy decoding (`do_sample=false`, `num_beams=1`), and a maximum of 128 new tokens. Each of the eight responses generated exactly 128 tokens and therefore hit the cap. The Qwen response for `topic_000` had already been generated before the prompt set was finalized and was reused because it had the same saved input; the other three Qwen responses and all four hybrid responses were generated in the final comparison pass. Models were loaded and run sequentially, not concurrently.

The Windows run used CUDA device 0 (RTX 5090) for layers 0–15, CUDA device 1 (RTX 3090) for layers 16–27, and CPU offload for layers 28–39 and the remaining listed components. The memory caps were 28 GiB on the 5090 and 20 GiB on the 3090. Optimized linear-attention kernels were unavailable in this environment, so the run used PyTorch reference kernels. CPU offload and reference kernels are important context for the absolute generation times.

## Runtime and output comparison

Generation time below is the sum of the recorded per-prompt generation durations; it excludes model loading. Throughput is total generated tokens divided by that summed generation time. The mean is across four prompts, with 128 generated tokens per prompt.

| Condition | Prompts | Generated tokens | Generation time total | Mean per prompt | Aggregate output rate | Load time recorded for pass |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 reference | 4 | 512 | 2,639.02 s (43.98 min) | 659.76 s | 0.1940 tokens/s | 85.98 s for the resumed three-prompt pass; earlier load time not retained |
| Qwen + Ornith layers 16–19 | 4 | 512 | 3,151.52 s (52.53 min) | 787.88 s | 0.1625 tokens/s | 91.19 s |

The hybrid averaged **128.12 seconds more per prompt**, or **19.4% longer** than the Qwen reference. Its aggregate generated-token rate was about **16.3% lower**. Each row below compares one identical tokenized input. The reported output texts are the complete generated continuations; each stops at the 128-token limit.

| Topic | Qwen time | Hybrid time | Same generated token sequence? |
|---|---:|---:|---|
| Autumn leaf pigments | 671.91 s | 787.09 s | No |
| REST API inventory search | 662.62 s | 793.77 s | No |
| Bus-lane evaluation | 650.72 s | 786.75 s | No |
| Encryption | 653.76 s | 783.90 s | No |

## Exact prompts and generated continuations

### `topic_000` — Autumn leaf pigments

**Prompt:** Explain why leaves change color in autumn. Describe the roles of chlorophyll, carotenoids, and anthocyanins, then explain how temperature and daylight affect the timing. Use plain language and distinguish what is well established from what varies by tree species.

**Qwen3.6 (128 tokens):**

~~~~text
Here is an explanation of why leaves change color in autumn, broken down by the science behind the pigments and the environmental triggers.

### 1. The Core Mechanism: Why Leaves Change Color
Leaves change color not because new colors are "created" in autumn, but because the **green pigment disappears**, revealing colors that were already there or newly produced.

In spring and summer, leaves are green because they are busy making food (sugar) through **photosynthesis**. This process requires a green pigment called **chlorophyll**. Chlorophyll is dominant and masks other pigments present in the leaf.

As autumn
~~~~

**Hybrid (128 tokens):**

~~~~text
Here is an explanation of why leaves change color in autumn, broken down by the science involved, the specific pigments, and the environmental factors.

### 1. The Core Reason: Why Leaves Change Color

Leaves change color in autumn because **chlorophyll production stops and existing chlorophyll breaks down**.

During spring and summer, leaves are green because they are full of **chlorophyll**, the pigment that helps trees convert sunlight into energy (photosynthesis). Chlorophyll is also the most dominant pigment, masking other colors that are present in the leaf but hidden by the green.

In autumn, as days shorten
~~~~

### `topic_010` — REST API inventory search

**Prompt:** Design a REST API endpoint for searching product inventory by category, minimum stock, and a free-text query. Specify the HTTP method, path, query parameters, example response, validation behavior, and how pagination should work.

**Qwen3.6 (128 tokens):**

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

**Hybrid (128 tokens):**

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
| `category` | string | Optional | Filter by product category (e.g., "electronics", "clothing") | Alphanumeric, hyphens, underscores only. Max 50 chars. |
| `min_stock` | integer | Optional | Minimum stock level to include in results | Integer ≥ 0. Max
~~~~

### `topic_025` — Bus-lane evaluation

**Prompt:** A city is considering a bus lane on a congested street. Outline an evaluation plan using travel time, reliability, passenger counts, safety, and effects on nearby streets. Explain how to compare results fairly before and after the change.

**Qwen3.6 (128 tokens):**

~~~~text
Here is a comprehensive evaluation plan for assessing the impact of a new bus lane on a congested street. This plan focuses on quantitative data collection, comparative analysis, and fair benchmarking.

### **1. Evaluation Plan Overview**
The evaluation will be conducted in three phases: **Baseline (Pre-Implementation)**, **Implementation (Short-term, 1–3 months)**, and **Stabilization (Long-term, 6–12 months)**.

#### **A. Key Metrics & Data Collection Methods**

| Metric | Definition | Data Collection Method |
| :--- | :--- | :--- |
| **1
~~~~

**Hybrid (128 tokens):**

~~~~text
Here is a comprehensive evaluation plan for assessing the impact of a new bus lane on a congested street. This plan focuses on quantitative data collection, comparative analysis, and fair benchmarking.

### **1. Evaluation Plan Overview**
The evaluation will be structured into **Pre-Implementation (Baseline)** and **Post-Implementation (Impact)** phases. Data will be collected over a minimum of 3 months for each phase to account for seasonal variations, weather patterns, and behavioral adaptation periods.

#### **A. Key Metrics & Data Collection Methods**

| Metric | Definition | Data Collection Method |
| :--- | :--- |
~~~~

### `topic_045` — Symmetric and asymmetric encryption

**Prompt:** Explain the difference between symmetric and asymmetric encryption, and why real secure protocols often use both. Include the role of a session key and authenticated encryption.

**Qwen3.6 (128 tokens):**

~~~~text
To understand modern cryptography, one must distinguish between the two primary methods of encryption and understand how they are combined to create secure communication channels (such as HTTPS/TLS).

Here is the step-by-step logical deduction of the differences, the necessity of combining them, and the specific roles of session keys and authenticated encryption.

### 1. Symmetric vs. Asymmetric Encryption

The fundamental difference lies in the number of keys used and the mathematical operations involved.

#### Symmetric Encryption
*   **Mechanism:** Uses a **single shared key** for both encryption and decryption.
*   **Analogy:** A physical padlock
~~~~

**Hybrid (128 tokens):**

~~~~text
To understand modern cryptography, one must understand the trade-off between **speed** and **key distribution**.

Here is a detailed explanation of the differences between symmetric and asymmetric encryption, why they are combined, and the specific roles of session keys and authenticated encryption.

---

### 1. Symmetric Encryption
**Definition:** Symmetric encryption uses a **single shared key** for both encrypting and decrypting data. The sender and receiver must possess the exact same secret key.

*   **Mechanism:** $Ciphertext = Encrypt(Plaintext, Key)$
*   **Common Algorithms:** AES (Advanced Encryption Standard),
~~~~

## Interpretation

The hybrid successfully loaded, used the substituted blocks, and completed all four free-running generations. The generated token sequence differed from the Qwen reference on all four prompts. The displayed completions share portions of their structure and wording, but diverge in details such as explanations, phrasing, and table formatting. With only one deterministic run per prompt, the experiment does not measure output variability.

The simple runtime comparison favors Qwen: the hybrid was slower on every prompt in this small set, by about 17.1% to 20.9% per prompt. This is an observed result for this hardware, software, and device placement. It does not show that layer substitution intrinsically adds 19.4% latency in other deployments; CPU offload, reference kernels, and runtime variation all affect these measurements.

## Limitations

- Four prompts are a small, topic-selected sample, not a broad language or task-quality evaluation.
- All eight outputs were cut off at the 128-token generation limit. No rubric-based correctness, completeness, or usefulness score was collected.
- This was a direct, untrained swap of four entire blocks from checkpoints with different training lineages. The experiment cannot isolate expert compatibility from the substituted attention, normalization, or router changes.
- Matching tensor shapes and tokenizer vocabulary are structural compatibility checks only. They do not show that hidden-state representations or downstream layer behavior are compatible.
- The comparison used one greedy continuation per prompt and no repeated seeds, perplexity measurement, benchmark score, or statistical uncertainty estimate.
- Both models ran in BF16, so this comparison does not measure quantized inference.
- The measured throughput is specific to the Windows setup, two-GPU placement, CPU offload, and fallback linear-attention kernels; it is not a router-only latency measurement.
