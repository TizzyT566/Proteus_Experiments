# Same-Prompt BF16 Comparison: Qwen, Ornith, and Frankenstein

**Experiment completed:** 2026-10-10 14:21:24 America/New_York (EDT), converted from the recorded completion time 2026-10-10T18:21:24Z.

**Outcome:** All three BF16 checkpoints loaded and generated on the same autumn-leaf prompt. Qwen and Frankenstein each produced 1,024 tokens and stopped at the configured cap without EOS. Ornith produced 818 tokens and ended with EOS. Qwen and Frankenstein had nearly equal measured generation rates; Frankenstein took 35.48 seconds longer for its 1,024-token continuation (0.58% in this run). This is a one-prompt generation comparison, not a quality benchmark.

## Objective

Compare one identical prompt across the Qwen checkpoint, the Ornith-1.5 checkpoint, and a Frankenstein hybrid made by replacing Qwen decoder layers 16-19 with Ornith layers. The purpose was to inspect longer saved continuations from all three BF16 models after the earlier four-prompt comparison had capped every answer at 128 tokens. The generation cap for this follow-up was 1,024 new tokens per checkpoint.

## Checkpoints and hybrid construction

| Role | Checkpoint | Revision | Precision |
|---|---|---|---|
| Qwen reference and hybrid backbone | `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` | BF16 |
| Layer source | `ornith-ai/Ornith-1.5-35B-A3B` | `10fbf86fed7ecee4a061f8b499a618f46001cac1` | BF16 |
| Hybrid | Qwen backbone with Ornith decoder layers 16, 17, 18, and 19 | Built from the two revisions above | BF16 |

The Qwen checkpoint is post-trained, not a Base checkpoint. The hybrid replaces four complete decoder blocks: attention, input and post-attention normalization, MoE router/gate, routed experts, and shared-expert tensors. The layer-swap preflight found matching tensor names and shapes for all 69 replaced tensors. It is an untrained cross-checkpoint swap; no adaptation was applied to align hidden-state representations across the block boundaries.

## Prompt and evaluation protocol

This run evaluated one fixed item, `topic_000` (autumn leaf pigments), selected from a four-prompt broad-topic set. There was no train/test split or random sampling in this comparison. The exact same prompt string was used for all three models:

> Explain why leaves change color in autumn. Describe the roles of chlorophyll, carotenoids, and anthocyanins, then explain how temperature and daylight affect the timing. Use plain language and distinguish what is well established from what varies by tree species.

Each model used its own local tokenizer and chat template, with the prompt as a user message and a generation prompt appended. `enable_thinking=False` was passed when the tokenizer supported it. Each formatted prompt contained 65 input tokens. The prompt text was identical; token IDs and template control tokens were not forced to be identical across checkpoint tokenizers.

All models ran sequentially, with each checkpoint released before loading the next. Generation used `torch.bfloat16`, CUDA BF16, evaluation/inference mode, greedy decoding (`do_sample=false`, `num_beams=1`), KV cache enabled, and `max_new_tokens=1024`. Generation stopped at EOS or at the token cap. TF32 was disabled. Automatic device mapping used GPU memory limits of 28 GiB and 20 GiB, plus a 400 GiB CPU limit. The recorded placement put decoder layers 0-15 on CUDA device 0, layers 16-27 on CUDA device 1, and layers 28-39 on CPU for each checkpoint. Optimized causal-convolution and flash-linear-attention kernels were unavailable, so fallback PyTorch kernels were used.

## Measured results

Generation time excludes checkpoint loading. Throughput is generated continuation tokens divided by generation seconds. The “load time” column is the separately recorded checkpoint load duration.

| Checkpoint | Prompt tokens | Generated tokens | Ended with EOS | Hit 1,024-token cap | Generation time | Load time | Generation rate |
|---|---:|---:|:---:|:---:|---:|---:|---:|
| Qwen BF16 | 65 | 1024 | No | Yes | 6077.88 s (101.30 min) | 84.91 s | 0.168480 tokens/s |
| Ornith 1.5 BF16 | 65 | 818 | Yes | No | 4855.27 s (80.92 min) | 384.14 s | 0.168477 tokens/s |
| Frankenstein BF16 hybrid | 65 | 1024 | No | Yes | 6113.35 s (101.89 min) | 84.74 s | 0.167502 tokens/s |


Qwen and Frankenstein each reached the cap at 1,024 tokens. Their generated token-ID sequences were not identical; the first differing generated token was at zero-based position 17. Frankenstein's generation took 35.48 seconds longer than Qwen's (0.58%), while its measured rate was 0.167502 tokens/s versus Qwen's 0.168480. Ornith ended naturally after 818 tokens, so its lower total generation time reflects a shorter continuation; its measured token rate was 0.168477 tokens/s.

## Output observations

- Qwen produced a structured explanation with sections for the three pigments and environmental cues. It stopped in the middle of its temperature section at the cap.
- Ornith produced a shorter structured answer, distinguished established pigment facts from debated anthocyanin functions, and reached EOS after its summary.
- Frankenstein also produced a structured explanation and reached the cap while beginning a comparison table. Its response shares some opening phrasing and section structure with Qwen but is not the same generated token sequence.

These are descriptive observations from one prompt. No human rubric, factuality score, completeness score, repeated sampling, or statistical uncertainty estimate was collected.

## Interpretation and limitations

For this prompt, the hybrid loaded and generated successfully. Its 1,024-token generation rate was within about 0.6% of Qwen's, despite substituting four complete decoder blocks. This result applies only to the recorded device placement and software environment; CPU offload and fallback kernels strongly affect absolute throughput. It does not measure router-only or expert-selection latency.

The two capped outputs are complete records of what the models generated, but they are not complete natural answers: neither ended with EOS. Ornith did end with EOS. The prompt used the same text for every condition, but each checkpoint applied its own tokenizer and chat template, which can affect the exact model input. A single hand-selected prompt cannot establish general response quality or compatibility, and the untrained swap changes attention, normalization, routing, and expert weights together. This run cannot isolate which swapped component caused any output difference.

## Exact generated continuations

The following text is the complete decoded continuation saved for each condition. Capped continuations intentionally end where the 1,024-token limit stopped generation.

### Qwen BF16 — 1024 tokens; EOS: no; cap reached: yes

~~~~text
Here is an explanation of why leaves change color in autumn, broken down by the science behind the pigments and the environmental triggers.

### 1. The Core Mechanism: Why Leaves Change Color
Leaves change color not because new colors are "created" in autumn, but because the **green pigment disappears**, revealing colors that were already there or newly produced.

In spring and summer, leaves are green because they are busy making food (sugar) through **photosynthesis**. This process requires a green pigment called **chlorophyll**. Chlorophyll is dominant and masks other pigments present in the leaf.

As autumn approaches, trees prepare for winter by shutting down photosynthesis. They form a special layer of cells at the base of each leaf stem (the petiole) that gradually blocks the flow of nutrients and water into the leaf. This process is called **abscission**. As chlorophyll breaks down and is not replaced, the green fades, revealing the other pigments.

---

### 2. The Roles of the Three Key Pigments

#### **Chlorophyll (The Green Mask)**
*   **Role:** Absorbs sunlight to power photosynthesis.
*   **Behavior in Autumn:** It is unstable and breaks down quickly when sunlight decreases. Trees actively reabsorb nitrogen and other nutrients from chlorophyll before the leaf falls.
*   **Result:** As chlorophyll disappears, the green color vanishes.

#### **Carotenoids (The Yellow and Orange)**
*   **Role:** These pigments help absorb light energy for photosynthesis and protect the leaf from damage by excess sunlight.
*   **Types:**
    *   **Carotenes:** Produce orange colors (e.g., carrots, sweet potatoes).
    *   **Xanthophylls:** Produce yellow colors.
*   **Behavior in Autumn:** Carotenoids are **always present** in the leaf but are masked by chlorophyll during the growing season. They are more stable than chlorophyll and break down slowly.
*   **Result:** When chlorophyll fades, carotenoids are revealed. This is why maples, hickories, and poplars often turn bright yellow or orange.
*   **Well-Established Fact:** The presence of carotenoids is universal in most deciduous leaves. The intensity of yellow/orange depends on how much chlorophyll was present and how quickly it breaks down.

#### **Anthocyanins (The Red and Purple)**
*   **Role:** These pigments are **not** present in most leaves during the growing season. They are **newly produced** in autumn.
*   **Function:** Scientists believe anthocyanins act as a "sunscreen" for the leaf, protecting it from excess light while the tree reabsorbs nutrients. They may also help deter insects or protect the leaf from freezing.
*   **Behavior in Autumn:** They are produced in the sap of the leaf when sugars become trapped (due to the closing abscission layer).
*   **Result:** This creates red, purple, and crimson hues.
*   **Species Variation:** Not all trees produce anthocyanins. For example:
    *   **High producers:** Maples (especially Red Maple), dogwoods, sumacs, and oaks (some species).
    *   **Low/No producers:** Birches, aspens, and hickories rely mostly on carotenoids (yellows) and do not produce significant anthocyanins.

---

### 3. How Temperature and Daylight Affect Timing

The timing and intensity of fall colors are driven by two main environmental factors: **day length** and **temperature**.

#### **Daylight (Photoperiod) – The Primary Trigger**
*   **Well-Established Fact:** The most reliable signal for trees to start preparing for winter is the **shortening of daylight hours** (decreasing photoperiod).
*   **Mechanism:** As days get shorter, trees detect this change and begin forming the abscission layer. This triggers the breakdown of chlorophyll.
*   **Consistency:** Daylight changes predictably every year, so the *start* of the color change is relatively consistent across years for a given location.

#### **Temperature – The Modifier**
*   **Well-Established Fact:** Temperature affects the **speed** of the process and the **vividness** of the colors, but it does not start the process.
*   **Cool Nights (40–50°F / 4–10°C):** Cool nights slow down the movement of sugars out of the leaf. This traps sugar in the leaf, which boosts anthocyanin production, leading to **brighter reds**.
*   **Warm Days:** Sunny days promote photosynthesis, producing more sugar, which also enhances
~~~~

### Ornith 1.5 BF16 — 818 tokens; EOS: yes; cap reached: no

~~~~text
# Why Leaves Change Color in Autumn

## The Short Version

Leaves contain several different pigments, but we usually only see green most of the year because green dominates. In autumn, the green fades and the other colors—yellows, oranges, and reds—become visible. It's really a story about pigments appearing, disappearing, or being newly made.

## The Three Main Pigments

**Chlorophyll (green)**

Chlorophyll is the pigment that leaves use to capture sunlight and make food through photosynthesis. It's green, and it's abundant, so it masks the other colors during the growing season.

*Well established:* Chlorophyll breaks down in autumn and is not always replaced. As it fades, the green color disappears.

**Carotenoids (yellow and orange)**

These pigments produce the yellows and oranges you see in leaves like aspens, ginkgo, and carrots. Importantly, carotenoids are present in the leaf all year—they're just hidden by the green chlorophyll.

*Well established:* When chlorophyll breaks down in autumn, the carotenoids were already there. We simply see them once the green fades. This is the most reliable part of the story.

**Anthocyanins (red, purple, and blue)**

This is where things get more interesting. Unlike carotenoids, anthocyanins are generally *not* present during the growing season. In autumn, some trees actively produce them, creating reds and purples in maples, dogwoods, and some oaks.

*Well established:* Anthocyanins are made fresh in autumn in certain species.
*Less certain / debated:* Their exact purpose is still discussed by scientists. Leading ideas include protecting leaves from too much sunlight (letting the tree recover more nutrients before the leaf falls) and acting as a kind of sunscreen or anti-freeze.

## How Daylight and Temperature Affect Timing

**Daylight (the main trigger)**

*Well established:* The biggest signal is the shortening of days in autumn. As daylight decreases, trees respond by forming a special layer of cells at the base of each leaf stem. This layer slowly cuts off the leaf, eventually letting it fall. The shortening day is the primary clock that tells trees it's time to change.

**Temperature (a modifier)**

*Varies and is more nuanced:* Temperature influences the *intensity* and *timing* of colors, but it works differently for different pigments:

- For **red anthocyanins**, warm sunny days and cool (but not freezing) nights tend to produce the most vivid reds. The warm light helps make sugar, and cool nights trap that sugar in the leaf, boosting color.
- For **carotenoids**, the effect is less dramatic and less consistent.
- Extreme cold, frost, or drought can cause leaves to drop early, cutting the color display short.

## What Varies by Tree Species

To summarize the key differences:

- **Carotenoids** show up in many species as simple "green fades to reveal yellow/orange."
- **Anthocyanins** are made by specific species (like maples) and require the right combination of conditions.
- Some **oaks** actually *retain* their brown/red colors through autumn rather than dropping to yellow, because they hold onto a compound called tannin.
- The exact timing of color change depends on local weather, water availability, and species, which is why the "peak" can arrive at different times in different places.

## Bottom Line

The core mechanism—chlorophyll fading to reveal carotenoids already present, plus some trees making fresh anthocyanins, all triggered mainly by shorter days—is well understood. The finer details, especially the role of temperature and the exact function of anthocyanins, vary by species and remain active areas of study.
~~~~

### Frankenstein BF16 hybrid — 1024 tokens; EOS: no; cap reached: yes

~~~~text
Here is an explanation of why leaves change color in autumn, broken down by the science involved, the specific pigments, and the environmental factors.

### 1. The Core Reason: Why Leaves Change Color

Leaves change color in autumn because **chlorophyll production stops and existing chlorophyll breaks down**.

During spring and summer, leaves are green because they are full of **chlorophyll**, the pigment that helps trees convert sunlight into energy (photosynthesis). Chlorophyll is also the most dominant pigment, masking other colors that are present in the leaf but hidden by the green.

In autumn, as days shorten and temperatures cool, trees prepare for winter. They form a special layer of cells at the base of each leaf stem (called the **abscission layer**). This layer acts like a cork, gradually blocking the flow of nutrients and water into the leaf. Without new chlorophyll being produced, the existing green pigment fades away, revealing the other colors that were there all along—or creating new ones.

---

### 2. The Roles of the Three Main Pigments

#### **Chlorophyll (The Green Mask)**
*   **Role:** Absorbs sunlight for photosynthesis.
*   **Autumn Behavior:** It is unstable and breaks down quickly when not replenished. As it disappears, the green color fades.
*   **Status:** **Well-established science.** All green plants contain chlorophyll.

#### **Carotenoids (The Yellow and Orange Hues)**
*   **Role:** These pigments help absorb light energy and protect the leaf from damage by excess sunlight. They are present in the leaf all summer long but are masked by chlorophyll.
*   **Autumn Behavior:** They are more stable than chlorophyll and break down more slowly. As the green fades, yellow and orange colors (like those in corn, carrots, and bananas) become visible.
*   **Status:** **Well-established science.** Carotenoids are responsible for yellow/orange colors in many common autumn trees (e.g., aspens, hickories, some maples).

#### **Anthocyanins (The Red and Purple Hues)**
*   **Role:** These pigments are **not** present in most leaves during summer. They are **produced in the fall** in some species. Scientists believe they may act as a "sunscreen" to protect the leaf from light damage, allowing the tree to recover more nutrients before the leaf falls, or they may deter insects.
*   **Autumn Behavior:** They are created when sugars get trapped in the leaf and react with sunlight. This produces bright reds, purples, and crimsons.
*   **Status:** **Well-established that they exist and are produced in fall.** However, **the exact biological purpose** (why trees invest energy in making them) is still debated among scientists. Some theories suggest they protect against cold stress or herbivores, but this varies by species and is not fully settled.

---

### 3. How Temperature and Daylight Affect Timing

Two main environmental cues trigger the color change:

#### **Daylight (Photoperiod) – The Primary Trigger**
*   **Mechanism:** Trees are extremely sensitive to the length of daylight. As days get shorter after the autumn equinox, trees detect this change and begin forming the abscission layer.
*   **Status:** **Well-established.** Daylight is the most consistent signal. This is why trees in the same latitude generally start changing color at similar times, regardless of local weather.

#### **Temperature and Weather – The Modifiers**
*   **Mechanism:** Once the daylight signal starts the process, weather determines how vibrant the colors are and how quickly they appear.
    *   **Cool (but not freezing) nights:** Enhance red color production by trapping sugars in the leaf.
    *   **Warm, sunny days:** Promote sugar production, which fuels anthocyanin (red) creation.
    *   **Drought or extreme cold:** Can cause leaves to drop early, resulting in duller colors or premature leaf fall.
*   **Status:** **Well-established that weather affects intensity and timing.** However, the **specific optimal temperatures** vary by tree species. For example:
    *   **Maples** often need cool nights and warm days for brilliant reds.
    *   **Birches** may turn yellow even in milder conditions.
    *   **Oaks** often turn brown rather than red or yellow, and their color change is less dependent on temperature extremes.

---

### 4. What Is Well-Established vs. What Varies by Species

| Factor | Well-Established Science | Varies by Tree Species |
|--------|--------------------------|------------------------|
| **Chlorophyll breakdown** | Always occurs in autumn in deciduous trees
~~~~
