import os
import sys
import argparse

# Set cache directories to local /tmp to avoid NFS/vLLM file lock crashes on the server
user = os.environ.get('USER', 'user')
os.environ["TRITON_CACHE_DIR"] = f"/tmp/{user}/triton_cache"
os.environ["VLLM_CONFIG_ROOT"] = f"/tmp/{user}/vllm_config"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/{user}/torch_cache"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Keep vLLM on default engine path (same as training)

import json
import pandas as pd
import torch
import re
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams

def run_evaluation(model_type: str, rl_adapter_name: str = "FinQA-RL-3epoch"):
    """
    model_type: "base", "sft", or "rl"
    """
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 1. Prepare Model Path
    if model_type == "base":
        model_path = MODEL_ID
    else:
        print(f"Preparing {model_type.upper()} model for evaluation...")
        # For SFT and RL, we need to merge the LoRA adapter with the base model first 
        # so vLLM can load it natively.
        
        SFT_ADAPTER_PATH = "./FinQA-SFT-finetuned/final_adapter"
        RL_ADAPTER_PATH  = f"./{rl_adapter_name}/final_adapter"

        if model_type == "sft":
            if not os.path.exists(SFT_ADAPTER_PATH):
                raise FileNotFoundError(f"SFT adapter not found at {SFT_ADAPTER_PATH}.")

            print(f"Loading base model and merging SFT adapter from {SFT_ADAPTER_PATH}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, device_map="cpu", torch_dtype=torch.float16
            )
            peft_model  = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
            merged_model = peft_model.merge_and_unload()

        else:  # rl
            # The RL LoRA was trained on top of the SFT-merged model.
            # We MUST replicate that base before loading the RL adapter,
            # otherwise we are applying RL delta weights to the wrong base.
            if not os.path.exists(SFT_ADAPTER_PATH):
                raise FileNotFoundError(f"SFT adapter not found at {SFT_ADAPTER_PATH}. Needed as RL base.")
            if not os.path.exists(RL_ADAPTER_PATH):
                raise FileNotFoundError(f"RL adapter not found at {RL_ADAPTER_PATH}. Did you finish training?")

            print(f"Step 1/2 — Loading base model and merging SFT adapter (RL base)...")
            base_model   = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, device_map="cpu", torch_dtype=torch.float16
            )
            sft_peft     = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
            sft_merged   = sft_peft.merge_and_unload()   # base + SFT weights

            print(f"Step 2/2 — Loading RL adapter from {RL_ADAPTER_PATH} and merging...")
            rl_peft      = PeftModel.from_pretrained(sft_merged, RL_ADAPTER_PATH)
            merged_model = rl_peft.merge_and_unload()    # base + SFT + RL weights

        # Save merged model to a temporary directory so vLLM can load it natively
        temp_dir = tempfile.mkdtemp()
        print(f"Saving merged model to {temp_dir}...")
        merged_model.save_pretrained(temp_dir)

        # Also save tokenizer so vLLM finds it in the same dir
        tokenizer_tmp = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer_tmp.save_pretrained(temp_dir)

        model_path = temp_dir

        # Free CPU RAM before loading vLLM on GPU
        del merged_model
        import gc
        gc.collect()

    print(f"\n--- Loading vLLM engine for {model_type.upper()} model ---")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True, # Disable CUDA graphs which can crash engine core initialization
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. Prepare Dataset
    print("\nLoading test dataset...")
    df = pd.read_json("Dataset/test.json")
    
    def expanded_table(table):
        """
        Formats a table as human-readable key-value pairs so the model can
        identify exact row/column intersections rather than positional guessing.
        Each cell is labelled: [ROW HEADER] [COL HEADER] = VALUE
        """
        if not table:
            return ''
        # First row = column headers
        col_headers = [cell.strip() for cell in table[0]]
        lines = []
        # Build a separator so the model can see columns clearly
        lines.append('  '.join(f'{h:>15}' for h in col_headers))
        lines.append('-' * max(60, 17 * len(col_headers)))
        for row in table[1:]:
            row_label = row[0].strip() if row else ''
            # Pipe-separated row for LLM readability
            cells = [cell.strip() for cell in row]
            lines.append('  '.join(f'{c:>15}' for c in cells))
            # Also emit explicit key-value pairs to help retrieval
            for col_idx, cell in enumerate(cells[1:], start=1):
                col_name = col_headers[col_idx] if col_idx < len(col_headers) else f'col{col_idx}'
                lines.append(f'  [{row_label}][{col_name}] = {cell}')
        return '\n'.join(lines) + '\n'

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
    
    # Helper to execute a single DSL step
    def execute_step(op, arg1, arg2):
        try:
            a = float(arg1)
            b = float(arg2)
            if op == 'add' or op == 'sum': return a + b
            if op == 'subtract': return a - b
            if op == 'multiply': return a * b
            if op == 'divide': return a / b if b != 0 else 0
            if op == 'max': return max(a, b)
            if op == 'min': return min(a, b)
            if op == 'average': return (a + b) / 2
            if op == 'exp': return a ** b
            if op == 'greater': return 'yes' if a > b else 'no'
        except Exception:
            return None
        return None

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        expected_str = str(gold_answers[i]).strip().lower()
        
        # Check formatting
        if "<steps>" in generated_text and "<Answer>" in generated_text:
            format_correct += 1
            
        # Parse and EXECUTE the generated DSL steps
        # Small language models cannot do floating point division in their head,
        # so relying on the <Answer> tag directly severely underestimates their true accuracy.
        steps_match = re.search(r"<steps>(.*?)</steps>", generated_text, re.DOTALL)
        is_correct = False
        
        if steps_match:
            lines = steps_match.group(1).strip().split('\n')
            results = {}
            last_res = None
            
            for step_idx, line in enumerate(lines):
                m = re.search(r'Step (\d+)\s*:\s*([a-zA-Z_\-]+)\((.*)\)', line)
                if not m: continue
                op = m.group(2).lower()
                args = [a.strip() for a in m.group(3).split(',')]
                
                resolved_args = []
                for arg in args:
                    if arg.startswith('#'):
                        ref = arg[1:]
                        if ref in results:
                            resolved_args.append(results[ref])
                    elif arg.startswith('const_'):
                        if arg == 'const_m1': resolved_args.append(-1.0)
                        else: resolved_args.append(float(arg.replace('const_','')))
                    else:
                        try:
                            resolved_args.append(float(arg.replace('%','').replace(',','')))
                        except ValueError:
                            pass
                
                if len(resolved_args) == 1:
                    resolved_args.append(resolved_args[0])
                    
                if len(resolved_args) >= 2:
                    res = execute_step(op, resolved_args[0], resolved_args[1])
                    if res is not None:
                        if len(resolved_args) > 2:
                            if op in ('sum', 'add'): res = sum(resolved_args)
                            elif op == 'average': res = sum(resolved_args) / len(resolved_args)
                        results[str(step_idx)] = res
                        last_res = res
            
            # Compare execution result with gold answer
            if last_res is not None:
                if expected_str in ('yes', 'no'):
                    if str(last_res).lower() == expected_str:
                        is_correct = True
                else:
                    try:
                        gold_f = float(expected_str)
                        if abs(float(last_res) - gold_f) <= 0.01 * max(abs(gold_f), 1e-4):
                            is_correct = True
                    except ValueError:
                        pass
        
        # Fallback to <Answer> tag if execution failed (e.g. for base model that doesn't use DSL)
        if not is_correct:
            answer_match = re.search(r"<Answer>(.*?)</Answer>", generated_text, re.DOTALL)
            if answer_match:
                try:
                    final_result_str = answer_match.group(1).strip().lower()
                    if expected_str in ("yes", "no") and final_result_str == expected_str:
                        is_correct = True
                    else:
                        final_result = float(final_result_str.rstrip('%'))
                        gold_f = float(expected_str)
                        if abs(final_result - gold_f) <= 0.01 * max(abs(gold_f), 1e-4) or abs(final_result / 100 - gold_f) <= 0.01 * max(abs(gold_f), 1e-4):
                            is_correct = True
                except ValueError:
                    pass
                    
        if is_correct:
            correct += 1

    accuracy = (correct / len(prompts)) * 100
    format_accuracy = (format_correct / len(prompts)) * 100
    
    print(f"\n======================================")
    print(f"RESULTS FOR: {model_type.upper()} MODEL")
    print(f"======================================")
    print(f"Execution Accuracy: {accuracy:.2f}% ({correct}/{len(prompts)})")
    print(f"Format Strictness:  {format_accuracy:.2f}% ({format_correct}/{len(prompts)})")
    print(f"======================================\n")

    # Save the results to a file for inspection
    df["generated_text"] = [output.outputs[0].text for output in outputs]
    output_filename = f"eval_results_{model_type}.json"
    df.to_json(output_filename, orient="records", indent=4)
    print(f"✅ Saved full generation results to: {output_filename}")
    print(f"======================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FinQA models")
    parser.add_argument("--model", type=str, default="rl", choices=["base", "sft", "rl"],
                        help="Which model to evaluate")
    parser.add_argument("--adapter-name", type=str, default="FinQA-RL-3epoch-v4",
                        help="Folder name of the RL run to evaluate (e.g. FinQA-RL-3epoch-v4)")
    args = parser.parse_args()

    run_evaluation(args.model, rl_adapter_name=args.adapter_name)
