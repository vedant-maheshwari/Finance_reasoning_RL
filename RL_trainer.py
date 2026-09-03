import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from reward_functions import format_reward, execution_reward, answer_tag_reward, grounding_reward, reasoning_reward
from trl import GRPOTrainer, GRPOConfig

# ── System prompt (must match reward_functions.SYSTEM_PROMPT) ──────────────────
SYSTEM_PROMPT = (
    "You are an expert financial reasoning assistant. "
    "Answer questions about financial reports precisely by following these steps:\n\n"

    "STEP 1 — CLASSIFY the question type:\n"
    "  • Percentage change / growth rate  → requires subtract then divide\n"
    "  • Percentage of a total / ratio     → requires divide only\n"
    "  • Absolute difference               → requires subtract only\n"
    "  • Sum or aggregation                → requires add or sum\n"
    "  • Comparison (greater/less)          → requires greater\n\n"

    "STEP 2 — RETRIEVE the exact values:\n"
    "  • Read every row and column header carefully before picking a number.\n"
    "  • Always prefer the 'Total' column or 'Total' row over sub-group columns (e.g. US-only).\n"
    "  • Use the value that directly answers the question, not the first number you see.\n\n"

    "STEP 3 — COMPUTE inside <steps>...</steps> using DSL format:\n"
    "  Format: Step N : operator(arg1, arg2)\n"
    "  Reference prior step results with #0, #1, #2, etc.\n"
    "  Available operators: add, subtract, multiply, divide, exp, greater, average, sum, max, min\n\n"

    "CRITICAL ORDERING RULES:\n"
    "  • Percentage change: Step 1: subtract(new_value, old_value), Step 2: divide(#0, old_value)\n"
    "  • Ratio / pct of total: Step 1: divide(part, whole) — do NOT subtract first\n"
    "  • Multi-step problems: chain ALL required steps; do not skip intermediate steps\n"
    "  • If the question asks 'by what amount did X change', use subtract only\n"
    "  • If the question asks 'what percentage did X change', use subtract THEN divide\n\n"

    "STEP 4 — STATE the answer inside <Answer>...</Answer>:\n"
    "  • Numeric questions: output the raw number (e.g. 94, 0.1446, -0.034)\n"
    "  • Yes/No questions: output yes or no"
)

# 1. Load Tokenizer
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Base Model + SFT Adapter, then MERGE into base weights
#    This avoids vLLM/generation issues with unmerged adapters and
#    gives us a clean starting point for the RL LoRA.
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
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
merged_model = sft_model.merge_and_unload()   # ← merge SFT weights into base
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
    then format with the chat template. GRPO expects a 'prompt' column.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *example["messages"],   # user turn(s) from the dataset
    ]
    example["prompt"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return example


dataset = dataset.map(format_chat_template, num_proc=4)

# 5. Setup GRPO Trainer
RUN_NAME = "FinQA-RL-6epoch-v5"   # ← bumped: 6 epochs, 8 generations, CoT prompt

training_args = GRPOConfig(
    output_dir=f"./{RUN_NAME}",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50,              # ~1% of steps — avoids warmup_ratio compat issues
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=6,             # Bumped from 3 to 6 to fix multi-step truncations
    num_generations=8,              # Bumped from 4 to 8 to increase exploration
    generation_batch_size=8,
    max_completion_length=1024,     # increased from 512 to 1024 to allow very long multi-step chains
    beta=0.04,                      # KL penalty weight
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,
    use_cpu=(device.type == "cpu"),
    bf16=use_bf16,
    # vLLM: safe because SFT adapter is merged; fresh RL LoRA is synced each step
    use_vllm=True,
    vllm_gpu_memory_utilization=0.6,  # Decreased from 0.7 to fit into remaining VRAM
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward,        # up to 1.0 — correct XML structure
        execution_reward,     # up to 2.0 — correct DSL execution (scale-aware partial reward added)
        answer_tag_reward,    # up to 2.0 — correct value in <Answer> tag (matches evaluator)
        grounding_reward,     # up to 1.0 — penalizes hallucinated numbers in arguments
        reasoning_reward,     # up to 1.0 — checks if the program is logically valid/executable
    ],
    args=training_args,
    train_dataset=dataset,
)

print("Starting RL training...")
trainer.train()

print("Saving final RL model...")
trainer.save_model(f"./{RUN_NAME}/final_adapter")
print(f"Done! Adapter saved to ./{RUN_NAME}/final_adapter")
