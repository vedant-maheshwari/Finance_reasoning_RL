import json
import re

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

def analyze(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    correct = 0
    format_correct = 0
    
    yes_no_total = 0
    yes_no_correct = 0
    numeric_total = 0
    numeric_correct = 0

    execution_success = 0
    fallback_success = 0
    
    for sample in data:
        generated_text = sample.get('generated_text', '')
        qa = sample.get('qa', {})
        expected_str = str(qa.get('exe_ans', '')).strip().lower()
        
        is_yes_no = expected_str in ('yes', 'no')
        if is_yes_no:
            yes_no_total += 1
        else:
            numeric_total += 1
            
        if "<steps>" in generated_text and "<Answer>" in generated_text:
            format_correct += 1
            
        steps_match = re.search(r"<steps>(.*?)</steps>", generated_text, re.DOTALL)
        is_correct = False
        used_fallback = False
        
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
            
            if last_res is not None:
                if is_yes_no:
                    if str(last_res).lower() == expected_str:
                        is_correct = True
                        execution_success += 1
                else:
                    try:
                        gold_f = float(expected_str)
                        if abs(float(last_res) - gold_f) <= 0.01 * max(abs(gold_f), 1e-4):
                            is_correct = True
                            execution_success += 1
                    except ValueError:
                        pass
        
        if not is_correct:
            answer_match = re.search(r"<Answer>(.*?)</Answer>", generated_text, re.DOTALL)
            if answer_match:
                try:
                    final_result_str = answer_match.group(1).strip().lower()
                    if is_yes_no and final_result_str == expected_str:
                        is_correct = True
                        fallback_success += 1
                    else:
                        final_result = float(final_result_str.rstrip('%'))
                        gold_f = float(expected_str)
                        if abs(final_result - gold_f) <= 0.01 * max(abs(gold_f), 1e-4) or abs(final_result / 100 - gold_f) <= 0.01 * max(abs(gold_f), 1e-4):
                            is_correct = True
                            fallback_success += 1
                except ValueError:
                    pass
                    
        if is_correct:
            correct += 1
            if is_yes_no:
                yes_no_correct += 1
            else:
                numeric_correct += 1

    print(f"--- Analysis of {file_path} ---")
    print(f"Total Samples: {len(data)}")
    print(f"Execution Accuracy (Overall): {correct/len(data)*100:.2f}% ({correct}/{len(data)})")
    print(f"Format Strictness: {format_correct/len(data)*100:.2f}% ({format_correct}/{len(data)})")
    if yes_no_total > 0:
        print(f"Yes/No Accuracy: {yes_no_correct/yes_no_total*100:.2f}% ({yes_no_correct}/{yes_no_total})")
    if numeric_total > 0:
        print(f"Numeric Accuracy: {numeric_correct/numeric_total*100:.2f}% ({numeric_correct}/{numeric_total})")
    print(f"Correct via Execution: {execution_success}")
    print(f"Correct via Fallback (Answer tag only): {fallback_success}")

if __name__ == '__main__':
    import sys
    analyze(sys.argv[1])
