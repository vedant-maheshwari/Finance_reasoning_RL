import re


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

# ── DSL constants used in FinQA programs ─────────────────────────────────────
# e.g. multiply(#0, const_100) means "multiply by 100 to convert fraction→percent"
_CONSTANTS = {
    "const_m1":         -1,
    "const_0":           0,
    "const_1":           1,
    "const_2":           2,
    "const_3":           3,
    "const_4":           4,
    "const_5":           5,
    "const_6":           6,
    "const_7":           7,
    "const_8":           8,
    "const_9":           9,
    "const_10":         10,
    "const_100":       100,
    "const_1000":     1_000,
    "const_10000":   10_000,
    "const_100000": 100_000,
    "const_1000000":         1_000_000,
    "const_1000000000": 1_000_000_000,
}


def _parse_arg(arg: str, intermediate_results: list) -> float:
    """
    Resolve a single DSL argument to a float.

    Handles:
      - #N            → look up intermediate result
      - const_X       → resolve via _CONSTANTS table
      - 23.6%         → strip % and divide by 100 → 0.236
      - plain floats  → direct conversion
    """
    arg = arg.strip()
    if arg.startswith("#"):
        return intermediate_results[int(arg[1:])]
    if arg.lower() in _CONSTANTS:
        return float(_CONSTANTS[arg.lower()])
    if arg.endswith("%"):
        return float(arg[:-1]) / 100.0
    return float(arg)


def execute_step(operator: str, args: list, intermediate_results: list):
    """
    Execute one DSL step and return the result (float or 'yes'/'no').

    Supported operators
    -------------------
    add, subtract, multiply, divide, exp        — binary arithmetic
    greater                                     — binary comparison → 'yes'/'no'
    average, sum, max, min                      — variadic aggregation (≥1 args)
    """
    operator = operator.strip().lower()
    parsed = [_parse_arg(a, intermediate_results) for a in args if a.strip()]

    if not parsed:
        raise ValueError("No arguments parsed")

    if operator == "add":
        return sum(parsed)                          # variadic: add(a, b, c, ...)
    elif operator == "subtract":
        return parsed[0] - parsed[1]
    elif operator == "multiply":
        result = 1.0
        for p in parsed:
            result *= p
        return result
    elif operator == "divide":
        if parsed[1] == 0:
            raise ZeroDivisionError("Division by zero")
        return parsed[0] / parsed[1]
    elif operator == "exp":
        return parsed[0] ** parsed[1]
    elif operator == "greater":
        return "yes" if parsed[0] > parsed[1] else "no"
    elif operator == "average":
        return sum(parsed) / len(parsed)
    elif operator == "sum":
        return sum(parsed)
    elif operator == "max":
        return max(parsed)
    elif operator == "min":
        return min(parsed)
    else:
        raise ValueError(f"Unknown operator: {operator}")


# ── Reward functions ──────────────────────────────────────────────────────────

def format_reward(completions, **kwargs):
    """
    Checks if the model used the correct XML tags and structure.
    Returns a score in [0.0, 1.0].
    """
    rewards = []
    for completion in completions:
        score = 0.0
        
        # 1. Base Structure (Max 0.5)
        if "<steps>" in completion and "</steps>" in completion:
            score += 0.2
        if "<Answer>" in completion and "</Answer>" in completion:
            score += 0.3
            
        # 2. Strict DSL Format (Max 0.5)
        # We penalize lines inside <steps> that are raw math instead of DSL
        steps_match = re.search(r"<steps>(.*?)</steps>", completion, re.DOTALL)
        if steps_match:
            steps_text = steps_match.group(1).strip()
            # extract non-empty lines
            lines = [l.strip() for l in steps_text.split('\n') if l.strip()]
            if lines:
                correct_lines = 0
                for line in lines:
                    # e.g., "Step 1 : add(1.5, 1.4)"
                    if re.match(r"^Step \d+\s*:\s*[a-zA-Z_\-]+\(.*\)$", line):
                        correct_lines += 1
                
                strictness_ratio = correct_lines / len(lines)
                score += (strictness_ratio * 0.5)
                
        rewards.append(score)
    return rewards


def execution_reward(completions, gold_answer, **kwargs):
    """
    Executes the model's DSL program and compares the result to the gold answer.

    Returns 2.0 for correct, 0.0 otherwise.
    Handles: const_X, %-args, average/max/min/sum, yes/no from greater().

    NOTE: grounding_reward has been removed — it saturated to mean=1.0, std=0
    in epoch 1 and provided zero gradient signal from that point on.
    """
    rewards = []

    for completion, expected_answer in zip(completions, gold_answer):
        try:
            # 1. Extract steps block
            steps_match = re.search(r"<steps>(.*?)</steps>", completion, re.DOTALL)
            if not steps_match:
                rewards.append(0.0)
                continue

            steps_lines = steps_match.group(1).strip().split("\n")

            # 2. Execute each step
            intermediate_results = []
            for step_line in steps_lines:
                m = re.search(r"Step \d+\s*:\s*([a-zA-Z_\-]+)\((.*)\)", step_line)
                if not m:
                    continue
                operator = m.group(1).strip()
                args     = m.group(2).split(",")
                result   = execute_step(operator, args, intermediate_results)
                intermediate_results.append(result)

            # 3. Guard: nothing executed
            if not intermediate_results:
                rewards.append(0.0)
                continue

            final_result = intermediate_results[-1]

            # 4. Compare to gold
            if isinstance(final_result, str):
                # yes/no from greater()
                rewards.append(
                    2.0 if final_result.lower() == str(expected_answer).strip().lower()
                    else 0.0
                )
            else:
                try:
                    expected  = float(expected_answer)
                    abs_diff  = abs(final_result - expected)
                    rel_tol   = 0.01 * max(abs(expected), 1e-4)  # 1% relative tolerance
                    
                    if abs_diff <= rel_tol:
                        rewards.append(2.0)
                    else:
                        # Check for scale partial reward (0.5 instead of 2.0)
                        scale_matched = False
                        for scale in [1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9]:
                            scaled_result = final_result * scale
                            if abs(scaled_result - expected) <= 0.01 * max(abs(expected), 1e-4):
                                scale_matched = True
                                break
                        
                        if scale_matched:
                            rewards.append(0.5) # Math correct, scale wrong
                        else:
                            rewards.append(0.0)
                except (ValueError, TypeError):
                    # gold is a string (yes/no) but result is numeric — mismatch
                    rewards.append(0.0)

        except Exception:
            rewards.append(0.0)

    return rewards


def answer_tag_reward(completions, gold_answer, **kwargs):
    """
    Directly rewards the model for putting the correct answer inside the
    <Answer>...</Answer> tag — this is exactly what the evaluator checks.

    Handles:
      - yes/no string answers (direct match, case-insensitive)
      - numeric answers with 1% relative tolerance
      - % suffix: '14.46%' accepted when gold is 0.1446 (strips %, divides by 100)

    Weight = 2.0 to match execution_reward and strongly reinforce the final answer.
    """
    rewards = []

    for completion, expected_answer in zip(completions, gold_answer):
        m = re.search(r"<Answer>(.*?)</Answer>", completion, re.DOTALL)
        if not m:
            rewards.append(0.0)
            continue

        try:
            predicted    = m.group(1).strip().lower()
            expected_str = str(expected_answer).strip().lower()

            # Yes/No
            if expected_str in ("yes", "no"):
                rewards.append(2.0 if predicted == expected_str else 0.0)
                continue

            # Numeric
            expected_f = float(expected_answer)
            rel_tol    = 0.01 * max(abs(expected_f), 1e-4)

            # Try direct parse
            try:
                predicted_f = float(predicted)
            except ValueError:
                # Try stripping % suffix
                if predicted.endswith("%"):
                    predicted_f = float(predicted[:-1]) / 100.0
                else:
                    rewards.append(0.0)
                    continue

            abs_diff = abs(predicted_f - expected_f)
            
            if abs_diff <= rel_tol:
                rewards.append(2.0)
            else:
                # Check for scale partial reward
                scale_matched = False
                for scale in [1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9]:
                    if abs((predicted_f * scale) - expected_f) <= rel_tol:
                        scale_matched = True
                        break
                rewards.append(0.5 if scale_matched else 0.0)

        except Exception:
            rewards.append(0.0)

    return rewards


def grounding_reward(completions, prompts, **kwargs):
    """
    Extracts all raw numbers used in the arguments of the generated steps,
    and checks if they exist in the raw prompt (context/table).
    Variables (#N), constants (const_X), and percentages are ignored.
    Returns 1.0 if all numbers are grounded, or a fractional reward.
    """
    rewards = []
    for completion, p_str in zip(completions, prompts):
        steps_match = re.search(r"<steps>(.*?)</steps>", completion, re.DOTALL)
        if not steps_match:
            rewards.append(0.0)
            continue
            
        lines = steps_match.group(1).strip().split("\n")
        extracted_numbers = []
        
        for line in lines:
            m = re.search(r"Step \d+\s*:\s*[a-zA-Z_\-]+\((.*)\)", line)
            if m:
                args = m.group(1).split(",")
                for arg in args:
                    arg = arg.strip()
                    # Ignore empty, refs, constants, percentages
                    if not arg or arg.startswith("#") or arg.startswith("const_") or arg.endswith("%"):
                        continue
                    try:
                        extracted_numbers.append(str(float(arg)))
                    except ValueError:
                        pass
        
        if not extracted_numbers:
            # FIX: If the program is valid but didn't need raw numbers (e.g. only consts/percentages), 
            # don't falsely penalize it! Give it full grounding reward.
            rewards.append(1.0) 
            continue
            
        # Check grounding
        grounded_count = 0
        # FIX: Financial docs have commas (e.g. 1,000). The model generates 1000.
        # We must strip commas from the prompt text so "1000" matches "1,000".
        prompt_text = p_str.lower().replace(",", "") 
        for num_str in extracted_numbers:
            # We check if the exact string or float format exists in the prompt
            # Many times floats like 4.0 appear as 4 in text.
            if num_str in prompt_text or num_str.rstrip("0").rstrip(".") in prompt_text:
                grounded_count += 1
                
        rewards.append(grounded_count / len(extracted_numbers))
        
    return rewards


def reasoning_reward(completions, **kwargs):
    """
    Checks if the generated step-by-step program executes without throwing errors.
    This gives the model partial credit (1.0) for learning how to write logically sound
    DSL, even if the final answer is wrong.
    """
    rewards = []
    for completion in completions:
        steps_match = re.search(r"<steps>(.*?)</steps>", completion, re.DOTALL)
        if not steps_match:
            rewards.append(0.0)
            continue
            
        steps_lines = steps_match.group(1).strip().split("\n")
        intermediate_results = []
        valid_execution = True
        has_steps = False
        
        for step_line in steps_lines:
            m = re.search(r"Step \d+\s*:\s*([a-zA-Z_\-]+)\((.*)\)", step_line)
            if not m:
                continue
            has_steps = True
            operator = m.group(1).strip()
            args = m.group(2).split(",")
            
            try:
                result = execute_step(operator, args, intermediate_results)
                intermediate_results.append(result)
            except Exception:
                valid_execution = False
                break
                
        if has_steps and valid_execution:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards
