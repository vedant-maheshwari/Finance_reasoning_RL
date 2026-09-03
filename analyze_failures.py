import json
import re
import random

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

def analyze_failures(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    failed_samples = []
    
    for sample in data:
        generated_text = sample.get('generated_text', '')
        qa = sample.get('qa', {})
        expected_str = str(qa.get('exe_ans', '')).strip().lower()
        is_yes_no = expected_str in ('yes', 'no')
        
        steps_match = re.search(r"<steps>(.*?)</steps>", generated_text, re.DOTALL)
        is_correct = False
        
        last_res = None
        generated_ops = []
        generated_args = []
        
        if steps_match:
            lines = steps_match.group(1).strip().split('\n')
            results = {}
            for step_idx, line in enumerate(lines):
                m = re.search(r'Step (\d+)\s*:\s*([a-zA-Z_\-]+)\((.*)\)', line)
                if not m: continue
                op = m.group(2).lower()
                args = [a.strip() for a in m.group(3).split(',')]
                
                generated_ops.append(op)
                
                resolved_args = []
                for arg in args:
                    if not arg.startswith('#') and not arg.startswith('const_'):
                        generated_args.append(arg)
                        
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
            
            if last_res is not None:
                if is_yes_no:
                    if str(last_res).lower() == expected_str:
                        is_correct = True
                else:
                    try:
                        gold_f = float(expected_str)
                        if abs(float(last_res) - gold_f) <= 0.01 * max(abs(gold_f), 1e-4):
                            is_correct = True
                    except ValueError:
                        pass
        
        if not is_correct:
            answer_match = re.search(r"<Answer>(.*?)</Answer>", generated_text, re.DOTALL)
            if answer_match:
                try:
                    final_result_str = answer_match.group(1).strip().lower()
                    if is_yes_no and final_result_str == expected_str:
                        is_correct = True
                    else:
                        final_result = float(final_result_str.rstrip('%'))
                        gold_f = float(expected_str)
                        if abs(final_result - gold_f) <= 0.01 * max(abs(gold_f), 1e-4) or abs(final_result / 100 - gold_f) <= 0.01 * max(abs(gold_f), 1e-4):
                            is_correct = True
                except ValueError:
                    pass
        
        if not is_correct:
            sample['generated_ops'] = generated_ops
            sample['generated_args'] = generated_args
            failed_samples.append(sample)

    print(f"Total Failed Samples: {len(failed_samples)}")
    
    # Categorize failures roughly
    # We will look at expected program vs generated program
    retrieval_failures = 0
    reasoning_failures = 0 # Right numbers, wrong ops
    
    for sample in failed_samples:
        expected_program = sample['qa'].get('program_re', '')
        gen_args = sample.get('generated_args', [])
        
        # Check if the expected numbers are in the generated text or args
        # This is a very rough heuristic
        steps = sample['qa'].get('steps', [])
        expected_args = []
        for step in steps:
            if 'arg1' in step:
                try: 
                    float(step['arg1'])
                    expected_args.append(step['arg1'])
                except: pass
            if 'arg2' in step:
                try: 
                    float(step['arg2'])
                    expected_args.append(step['arg2'])
                except: pass
                
        # Are expected args found in generated args?
        found_all = True
        for ea in expected_args:
            found = False
            for ga in gen_args:
                if ga == ea or (float(ga) if ga.replace('.','',1).isdigit() else None) == (float(ea) if ea.replace('.','',1).isdigit() else None):
                    found = True
                    break
            if not found:
                found_all = False
                break
                
        if not found_all and expected_args:
            retrieval_failures += 1
        else:
            reasoning_failures += 1
            
    print(f"Rough Heuristic Breakdown:")
    print(f"  Retrieval/Extraction Failures (Wrong Numbers Picked): {retrieval_failures}")
    print(f"  Reasoning/Operation Failures (Right Numbers, Wrong Math/Order): {reasoning_failures}")
    
    print("\n--- 5 Random Failure Examples ---")
    random.seed(42)
    for sample in random.sample(failed_samples, min(5, len(failed_samples))):
        print(f"Q: {sample['qa']['question']}")
        print(f"Expected Ans: {sample['qa']['exe_ans']}")
        print(f"Expected Prog: {sample['qa'].get('program_re', sample['qa'].get('program'))}")
        
        gen = sample.get('generated_text', '')
        steps_match = re.search(r"<steps>.*?</steps>", gen, re.DOTALL)
        if steps_match:
            print("Generated Steps:")
            print(steps_match.group(0))
        else:
            print("No generated steps found.")
        print("-" * 40)

if __name__ == '__main__':
    import sys
    analyze_failures(sys.argv[1])
