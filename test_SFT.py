import torch
from transformer import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel 

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LoRA_ADAPTER_PATH = "./FinQA-SFT-finetuned/final_adapter"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'you are running on {device}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM(MODEL_ID,
                                dtype=torch.bfloat16,
                                device_map='auto')

model = PeftModel.from_pretrained(base_model, LoRA_ADAPTER_PATH)

#test prompt
prompt = "You are a financial analyst.\n\nContext:\nRevenue in 2014 was $500. Revenue in 2015 was $600.\n\nQuestion: What is the total revenue?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

#generation
ouputs = model.generate(
    **inputs,
    max_new_tokens = 256,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.9,
    eos_token_id = tokenizer.eos_token_id,
    pad_token_id = tokenizer.pad_token_id
)

response = tokenizer.decode(ouputs[0], skip_special_tokens=True)
print(response)
        