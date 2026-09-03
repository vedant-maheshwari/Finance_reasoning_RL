# Finance Reasoning RL Fine-Tuning

This repository contains the codebase for fine-tuning Small Language Models (SLMs) to perform complex, multi-step numerical reasoning over financial documents (using the FinQA dataset).

## Overview

Large Language Models (LLMs) often struggle with numerical accuracy and logical consistency when parsing dense financial reports (e.g., SEC 10-K filings). This project proposes a Reinforcement Learning (RL) framework that optimizes reasoning correctness, numerical consistency, and evidence grounding beyond conventional Supervised Fine-Tuning (SFT).

The training pipeline consists of two phases:
1. **SFT Warmup:** Teaches the model the structural syntax (step-by-step reasoning programs).
2. **RL Fine-tuning:** Teaches the model to actually explore paths and generalize logic using a composite reward system.

## Setup Instructions

### 1. Install Requirements
Make sure you have an active Python virtual environment, then install the required packages:

```bash
pip install -r requiremnets.txt
```

*(Note: PyTorch will automatically use the `mps` backend on Apple Silicon MacBooks, or `cuda` on Nvidia GPUs).*

### 2. Dataset Preparation
The raw FinQA dataset needs to be formatted into standard Hugging Face conversational format (`messages` array) before it can be used for SFT.

Run the dataset preparator script:
```bash
python3 Dataset/SFT_dataset_preparator.py
```
*   **Input:** `Dataset/train.json`
*   **Output:** `SFT_train_formatted.json`
*   **Note:** By default, this script only formats 10% of the training data for the SFT warmup to prevent the model from memorizing the answers, leaving the remaining 90% for the actual RL phase.

### 3. Running Supervised Fine-Tuning (SFT)
Once the dataset is prepared, you can start the SFT warmup phase. The trainer uses `trl` and `peft` (LoRA) to efficiently fine-tune the model without running out of memory.

```bash
python3 SFT_trainer.py
```
*   **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct`
*   **Input Data:** `SFT_train_formatted.json`
*   **Output:** The trained adapter weights will be saved to the `./FinQA-SFT-finetuned` directory.

### 4. Running Reinforcement Learning (GRPO)
After the SFT adapter is trained, we use GRPO (via `trl`) on the remaining training data. We utilize a custom reward function that checks for exact answer matches and equation correctness.

```bash
python3 RL_trainer.py
```

## 🚀 Results: Achieving ~60% End-to-End Accuracy

Traditional Small Language Models (SLMs) fine-tuned on financial data often struggle with numerical consistency and complex reasoning. By implementing a **Reinforcement Learning (GRPO)** framework optimizing directly for **Execution Accuracy**, we guided the model to learn logical math deduction from scratch.

After scaling the training to 6 epochs and tuning the exploration paths (`num_generations=8`), the model reached a highly competitive **~60% Execution Accuracy**:

**Final RL Performance (v5):**
- **Overall Execution Accuracy (Exact Match):** **59.81%** (686 / 1147 correct)
- **Yes/No Question Accuracy:** **85.00%** (17 / 20 correct)
- **Format Strictness:** **99.83%** (Perfectly adhered to the reasoning DSL)

### 📊 Baseline Comparison

When the FinQA dataset was released, the state-of-the-art baseline was **FinQANet**. It relied on a dedicated Retriever model to spoon-feed the generator only the exact 1-2 sentences needed to solve the math.

| Model Setup | Architecture | Retriever Used? | Execution Accuracy |
| :--- | :--- | :---: | :---: |
| **Original FinQANet** | RoBERTa-Large (355M) | **Yes** | $\sim 61.24\%$ |
| **Our RL Model** | Qwen2.5 (1.5B) | **No (End-to-End)** | **59.81\%** |

### 🧠 Why is this significant?
Operating **End-to-End** without a retriever means our 1.5B model is fed the entire noisy financial report (the "haystack") and must act as both the retriever and the reasoning engine simultaneously. The original FinQA authors noted that attempting this End-to-End caused accuracy to plummet due to noise.

By achieving **59.81%**, our RL-trained SLM has essentially closed the gap with the Retriever-backed SOTA. This strongly validates that GRPO (Group Relative Policy Optimization) can effectively unlock advanced reasoning capabilities in small parameter models.

For an in-depth analysis of the RL progression, reward shaping, and a deep dive into the remaining failure modes, please read the full report: [`FinQA_RL_Results.md`](FinQA_RL_Results.md).