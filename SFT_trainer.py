import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = 'SFT_train_formatted.json'
OUTPUT_DIR = './FinQA-SFT-finetuned'

dataset = load_dataset('json', data_files=DATASET_PATH, split = "train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32,
    device_map = "auto"
)
# model.to(device)    

lora_config = LoraConfig(
    r=16,
    lora_alpha = 32,
    target_modules=['q_proj','v_proj'],
    lora_dropout = 0.05,
    bias = 'none',
    task_type = 'CAUSAL_LM'
)

model = get_peft_model(model, lora_config)

training_args = SFTConfig(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy='epoch',
    fp16=False,
    bf16=False,
    dataset_text_field="text",
    max_length=1024,
    loss_type="nll",
)

def format_chat_template(example):
    example['text'] = tokenizer.apply_chat_template(example['messages'],
                                                    tokenize=False,
                                                    add_generation_prompt=False)
    return example

dataset = dataset.map(format_chat_template)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    args=training_args
)

print("Starting SFT training ...")
trainer.train()

trainer.save_model(f'{OUTPUT_DIR}/final_adapter')
print("Training Completed!!")