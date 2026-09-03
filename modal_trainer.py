import modal

app = modal.App("finqa_rl_trainer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "datasets",
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "vllm",   # install latest stable; 0.25.1 does not exist and caused silent failures
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            ".git", "__pycache__",
            "Dataset/train.json", "Dataset/test.json",
            "FinQA-RL", "*.pdf", "*.ipynb",
            "RL_project", ".venv", "venv", "env",
        ],
    )
)

volume = modal.Volume.from_name("finqa_rl_results", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface_cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={
        "/output": volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    gpu="A100",
    timeout=60 * 60 * 12,   # 12 hours max
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_training():
    import os
    import sys
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType

    os.chdir("/root/project")
    sys.path.append("/root/project")

    from reward_functions import format_reward, execution_reward, answer_tag_reward
    from trl import GRPOTrainer, GRPOConfig

    # ── System prompt (must match reward_functions.SYSTEM_PROMPT) ────────────
    SYSTEM_PROMPT = (
        "You are a financial reasoning assistant. "
        "Solve the problem step by step inside <steps>...</steps> tags using the format: "
        "Step N : operator(arg1, arg2). Reference prior results with #0, #1, etc. "
        "State the final answer inside <Answer>...</Answer>. "
        "For numeric questions output a number; for comparison questions (greater/less) output yes or no."
    )

    # 1. Load Tokenizer
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load base model + SFT adapter, then MERGE into base weights
    #    This is required for stable GRPO generation — unmerged adapters
    #    cause silent failures when TRL tries to sync generation weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",  # PyTorch native SDPA (torch 2.0+, no extra package needed)
    )

    SFT_ADAPTER_PATH = "./FinQA-SFT-finetuned/final_adapter"
    print(f"Loading SFT adapter from {SFT_ADAPTER_PATH} and merging into base...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    merged_model = sft_model.merge_and_unload()   # ← merge SFT weights
    print("Merge complete.")

    # 3. Wrap merged model in a FRESH LoRA for RL training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(merged_model, lora_config)
    model.print_trainable_parameters()

    # 4. Load Dataset
    DATASET_PATH = "Dataset/RL_train_formatted.json"
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_chat_template(example):
        """
        Prepend a system prompt so the model knows the expected output format,
        then apply the chat template. GRPO expects a 'prompt' column.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *example["messages"],
        ]
        example["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return example

    dataset = dataset.map(format_chat_template, num_proc=4)

    # 5. Setup GRPO Trainer
    training_args = GRPOConfig(
        output_dir="/output/FinQA-RL-3epoch-v2",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        num_generations=4,              # 8→4: halves generation cost per step (~3.5s vs ~7.5s)
        generation_batch_size=4,
        max_completion_length=512,      # increased: 256 truncated multi-step DSL chains mid-way
        beta=0.04,                      # KL penalty weight
        save_steps=50,
        save_total_limit=3,
        logging_steps=10,
        use_cpu=(device.type == "cpu"),
        bf16=use_bf16,
        # vLLM: safe because SFT adapter is merged; fresh RL LoRA is synced each step
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward,        # up to 1.0 — correct XML structure
            execution_reward,     # up to 2.0 — correct DSL execution (const_X, %, avg/max/min/sum fixed)
            answer_tag_reward,    # up to 2.0 — correct value in <Answer> tag (matches evaluator)
            # grounding_reward removed: saturated to mean=1.0/std=0 in epoch 1, zero gradient signal
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting RL training...")
    trainer.train()

    print("Saving final RL model...")
    trainer.save_model("/output/FinQA-RL-3epoch-v2/final_adapter")

    # Commit the volume so changes persist beyond the container lifetime
    volume.commit()
    print("Done! Changes committed to Modal volume 'finqa_rl_results'.")


@app.local_entrypoint()
def main():
    print("Deploying and running training on Modal...")
    # Logs are streamed to your terminal automatically.
    # To run detached: modal run --detach modal_trainer.py
    run_training.remote()
