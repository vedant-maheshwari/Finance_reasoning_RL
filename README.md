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

### Next Steps (RL Phase)
After the SFT adapter is trained, the next phase of the project involves loading this adapter and using PPO/GRPO (via `trl`) on the remaining training data, utilizing a custom reward function that checks for exact answer matches and equation correctness.