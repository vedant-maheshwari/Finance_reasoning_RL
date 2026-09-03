import modal
import os
import sys

app = modal.App("finqa_evaluator")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "datasets",
        "transformers",
        "peft",
        "vllm",  # latest vllm
        "pandas"
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            ".git", "__pycache__", 
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
    timeout=60 * 60 * 2,   # 2 hours max for eval
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_evaluation(model_type: str):
    """
    model_type: "base", "sft", or "rl"
    """
    import os
    os.environ["VLLM_USE_V1"] = "0"  # Disable V1 engine which crashes on Modal's shm limits
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    import json
    import pandas as pd
    import torch
    import re
    import tempfile
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from vllm import LLM, SamplingParams
    
    os.chdir("/root/project")
    
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 1. Prepare Model Path
    if model_type == "base":
        model_path = MODEL_ID
    else:
        print(f"Preparing {model_type.upper()} model for evaluation...")
        # For SFT and RL, we need to merge the LoRA adapter with the base model first 
        # so vLLM can load it natively.
        adapter_path = "./FinQA-SFT-finetuned/final_adapter" if model_type == "sft" else "/output/FinQA-RL/final_adapter"
        
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found at {adapter_path}. Did you finish training?")
            
        print(f"Loading base model and merging with {adapter_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu", # Keep on CPU for merging to save VRAM
            torch_dtype=torch.float16
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = peft_model.merge_and_unload()
        
        # Save merged model to a temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f"Saving merged model to {temp_dir}...")
        merged_model.save_pretrained(temp_dir)
        
        # Also need to save tokenizer so vLLM finds it in the same dir
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.save_pretrained(temp_dir)
        
        model_path = temp_dir
        
        # Clear RAM
        del merged_model
        del peft_model
        del base_model
        import gc
        gc.collect()

    print(f"\n--- Loading vLLM engine for {model_type.upper()} model ---")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.6, # Lowered to avoid OOM if memory is fragmented
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True, # Disable CUDA graphs which can crash engine core initialization
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. Prepare Dataset
    print("\nLoading test dataset...")
    df = pd.read_json("Dataset/test.json")
    
    def expanded_table(table):
        formatted_str = ''
        for row in table:
            for cell in row:
                formatted_str+=cell+'|'
            formatted_str += '\n'
        return formatted_str

    SYSTEM_PROMPT = (
        "You are a financial reasoning assistant. "
        "Solve the problem step by step inside <steps>...</steps> tags using the format: "
        "Step N : operator(arg1, arg2). Reference prior results with #0, #1, etc. "
        "State the final numeric answer inside <Answer>...</Answer>."
    )

    prompts = []
    gold_answers = []
    
    for _, sample in df.iterrows():
        pre_text = ''.join(sample['pre_text'])
        table = sample['table']
        post_text = ''.join(sample['post_text'])
        question = sample['qa']['question']
        gold_answers.append(sample['qa']['exe_ans'])
        
        user_content = f"{pre_text} \n\n {expanded_table(table)} \n\n {post_text} \n\n {question} \n\n"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    print(f"Total test samples: {len(prompts)}")

    # 3. Generate
    sampling_params = SamplingParams(
        temperature=0.0, # Greedy search for evaluation
        max_tokens=512,
        stop=[tokenizer.eos_token]
    )
    
    print("\nStarting generation...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 4. Evaluate
    print("\nEvaluating results...")
    correct = 0
    format_correct = 0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        expected = float(gold_answers[i])
        
        # Check formatting
        if "<steps>" in generated_text and "<Answer>" in generated_text:
            format_correct += 1
            
        # Extract answer
        answer_match = re.search(r"<Answer>(.*?)</Answer>", generated_text, re.DOTALL)
        if answer_match:
            try:
                final_result = float(answer_match.group(1).strip())
                # Use identical tolerance as execution_reward
                abs_diff = abs(final_result - expected)
                rel_tol = 0.01 * max(abs(expected), 1e-4)
                if abs_diff <= rel_tol:
                    correct += 1
            except ValueError:
                pass
        else:
            # Fallback if model didn't use tags (mainly for base model)
            # Try to extract the very last number in the text
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
            if numbers:
                try:
                    final_result = float(numbers[-1])
                    abs_diff = abs(final_result - expected)
                    rel_tol = 0.01 * max(abs(expected), 1e-4)
                    if abs_diff <= rel_tol:
                        correct += 1
                except ValueError:
                    pass

    accuracy = (correct / len(prompts)) * 100
    format_accuracy = (format_correct / len(prompts)) * 100
    
    print(f"\n======================================")
    print(f"RESULTS FOR: {model_type.upper()} MODEL")
    print(f"======================================")
    print(f"Execution Accuracy: {accuracy:.2f}% ({correct}/{len(prompts)})")
    print(f"Format Strictness:  {format_accuracy:.2f}% ({format_correct}/{len(prompts)})")
    print(f"======================================\n")

@app.local_entrypoint()
def main(model: str = "base"):
    if model not in ["base", "sft", "rl"]:
        print("Error: model must be one of: 'base', 'sft', 'rl'")
        sys.exit(1)
        
    print(f"Submitting evaluation job for {model} model...")
    run_evaluation.remote(model)
