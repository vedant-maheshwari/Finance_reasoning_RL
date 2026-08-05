import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from reward_functions import format_reward, grounding_reward, execution_reward
from trl import GRPOTrainer, GRPOConfig

# 1. Load Tokenizer
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Base Model and SFT Adapter
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32,
    device_map="auto"
)

SFT_ADAPTER_PATH = "./FinQA-SFT-finetuned/final_adapter"
print(f"Loading SFT Adapter from {SFT_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

# 3. Load Dataset
DATASET_PATH = 'Dataset/RL_train_formatted.json'
print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset('json', data_files=DATASET_PATH, split="train")

def format_chat_template(example):
    # GRPO expects a 'prompt' column which contains the fully formatted string
    example['prompt'] = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=True)
    return example

dataset = dataset.map(format_chat_template)

# 4. Setup GRPO Trainer
training_args = GRPOConfig(
    output_dir="./FinQA-RL",
    learning_rate=1e-5,
    per_device_train_batch_size=1, # Keep tiny for 1.5B model + 8 generations
    gradient_accumulation_steps=4,
    num_generations=8,             # Generate 8 answers per question
    generation_batch_size=8,
    max_completion_length=256,
    beta=0.04,                     # KL penalty weight
    use_cpu=(device.type == "cpu"),
    bf16=(device.type == "cuda" and torch.cuda.is_bf16_supported()),
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, grounding_reward, execution_reward],
    args=training_args,
    train_dataset=dataset,
)

print("Starting RL training...")
trainer.train()

print("Saving final RL model...")
trainer.save_model("./FinQA-RL/final_adapter")
print("Done!")
