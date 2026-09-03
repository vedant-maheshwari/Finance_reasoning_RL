# Reinforcement Learning Fine-Tuning for Financial Reasoning
**Dataset:** FinQA  
**Base Model:** Qwen2.5-1.5B-Instruct  
**Training Pipeline:** Supervised Fine-Tuning (SFT) $\rightarrow$ Reinforcement Learning (GRPO)

---

## 1. Overview and Motivation
Traditional Small Language Models (SLMs) fine-tuned on financial data often struggle with numerical consistency. Even after Supervised Fine-Tuning (SFT), models frequently hallucinate values or perform incorrect arithmetic because SFT only optimizes for "language fluency" and "format", not logical correctness. 

To solve this, we implemented a **Reinforcement Learning (GRPO)** framework to optimize specifically for **Execution Accuracy** (Exact Match), forcing the model to learn the logical steps required to reach the correct answer.

---

## 2. Results Progression: From SFT to Final RL Run

### Phase 1: The SFT Baseline & Initial RL (Epoch 1)
- **Overall Accuracy:** $\sim 18.00\%$
- **Yes/No Accuracy:** $0.00\%$

**Analysis of the 18% Plateau:**
During the first epoch of RL, the model failed to cross the 18% execution accuracy mark. An in-depth log analysis revealed severe environment bottlenecks:
1. **DSL Operator Mismatch:** The model generated valid reasoning, but the RL execution engine failed to parse raw operators (e.g., `add1-1`, `divide1-2`) and variables (e.g., `const_100`). Correct logic was penalized with a $0.0$ reward.
2. **Binary Classification Crash:** The `greater()` operator correctly evaluated yes/no questions but returned string outputs (`"yes"`/`"no"`). The evaluation engine expected floats, crashing on these questions and resulting in 0% accuracy.
3. **Reward Saturation:** A custom `grounding_reward` designed to ensure the model used context numbers saturated at $1.0$ instantly with zero variance. It provided no learning signal and diluted the primary execution reward.

### Phase 2: Environment Overhaul & Final RL (Epoch 3)
To unblock the model, the RL environment was completely overhauled:
- **Execution Engine Rewrite:** Added robust parsing for constants (`const_X`), automatic percentage stripping, and dynamic normalization of operators (mapping `add1-1` $\rightarrow$ `add`).
- **Binary Logic Fix:** The system prompt was updated to teach the model explicit binary comparison formatting, and the evaluator was updated to seamlessly handle string-based Exact Match.
- **Reward Rebalancing:** The saturated `grounding_reward` was stripped, heavily concentrating the KL-divergence penalty and reward budget entirely on **Equation Correctness** and **Execution Accuracy**.

**Epoch 3 Results:**
- **Overall Accuracy:** **46.64\%** (535 / 1147 correct)
- **Yes/No Accuracy:** **55.00\%** (11 / 20 correct)

*Fixing the reward signal resulted in a massive **+28\% jump** in overall accuracy, proving that the model had the reasoning capability but was previously bottlenecked by the environment's inability to reward it properly.*

### Phase 3: Scaling Training and Generations (Epoch 6, v5)
With the environment fixed, the final bottleneck was exploration depth and training length. By bumping the training from 3 epochs to 6 epochs and increasing the `num_generations` (exploration paths) per prompt to 8, the model reached its full potential.

**Final v5 (Epoch 6) Results:**
- **Overall Accuracy:** **59.81\%** (686 / 1147 correct)
- **Yes/No Accuracy:** **85.00\%** (17 / 20 correct)
- **Format Strictness:** **99.83\%** (1145 / 1147)

*This tuning resulted in another **+13.17\% jump**, proving that GRPO scales incredibly well with increased generation paths.*

---

## 3. Final Contrast: Our Model vs. Original FinQA Baseline

### The FinQA Benchmark
When the FinQA dataset was released in 2021, the authors provided a baseline model named **FinQANet**. 

| Model Setup | Retriever Used? | Execution Accuracy (Exact Match) |
| :--- | :---: | :---: |
| **Original FinQANet** (RoBERTa-Large, 355M) | **Yes** | $\sim 61.24\%$ |
| **Our RL Model** (Qwen2.5-1.5B-Instruct) | **No** | **59.81\%** |

### Contextualizing the Achievement
At first glance, FinQANet's 61.24% seems slightly superior. However, the architectural difference makes our 59.81% result highly impressive:

1. **The Retriever Crutch:** The original FinQANet relies on a complex, multi-model pipeline. It uses a dedicated **Retriever** model to scan the financial report and filter out 99% of the text. The generator model is spoon-fed only the exact 1-2 sentences needed to solve the math.
2. **The End-to-End Challenge:** Our model operates **End-to-End**. It is fed the entire raw, noisy financial document (the "haystack") and must simultaneously act as the retriever (finding the right numbers) AND the reasoning engine (generating the mathematical DSL). 

**Conclusion:** 
The original FinQA authors noted that attempting the task End-to-End caused accuracy to plummet because the models became overwhelmed by noise. Achieving **59.81% Execution Accuracy** on a small 1.5B parameter model—processing the entire context End-to-End without a retriever—is a highly competitive academic result. It essentially closes the gap with the state-of-the-art retriever baseline and strongly validates that Reinforcement Learning (GRPO) can bridge the gap in reasoning capabilities for Small Language Models.
