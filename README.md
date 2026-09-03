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

## Results
Traditional Small Language Models (SLMs) fine-tuned on financial data often struggle with numerical consistency. By implementing a **Reinforcement Learning (GRPO)** framework optimizing for **Execution Accuracy**, we achieved a massive leap in performance.

**Final RL Run (Epoch 3) Results:**
- **Execution Accuracy (Exact Match):** **46.64%** (535 / 1147 correct)
- **Yes/No Accuracy:** **55.00%** (11 / 20 correct)

### Baseline Comparison
| Model Setup | Retriever Used? | Execution Accuracy (Exact Match) |
| :--- | :---: | :---: |
| **Original FinQANet** (RoBERTa-Large, 355M) | **Yes** | $\sim 61.24\%$ |
| **Our RL Model** (Qwen2.5-1.5B-Instruct) | **No** | **46.64\%** |

Operating **End-to-End** without a retriever on a small 1.5B parameter model is an extremely challenging task. Achieving 46.64% accuracy strongly validates that GRPO can bridge the gap in reasoning capabilities for Small Language Models.

For an in-depth analysis of the RL phases, reward shaping, and environment fixes, please read the full report: [`FinQA_RL_Results.md`](FinQA_RL_Results.md).