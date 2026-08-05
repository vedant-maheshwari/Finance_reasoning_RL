import re

def format_reward(completions, **kwargs):
    """
    Checks if the model used the correct XML tags and structure.
    Returns 1.0 for perfect format, 0.0 for completely broken format.
    """
    rewards = []
    
    # 'completions' is a list of strings (the model's generated answers)
    for completion in completions:
        score = 0.0
        
        # 1. Did it open the steps block?
        if "<steps>" in completion:
            score += 0.2
            
        # 2. Did it close the steps block?
        if "</steps>" in completion:
            score += 0.2
            
        # 3. Did it use the Step N : format?
        if re.search(r"Step \d+ :", completion):
            score += 0.2
            
        # 4. Did it provide an Answer block?
        if "<Answer>" in completion and "</Answer>" in completion:
            score += 0.4
            
        rewards.append(score)
        
    return rewards

def grounding_reward(completions, gold_context, **kwargs):
    """
    Checks if the numbers used in the steps actually exist in the context text.
    'gold_context' is passed from your dataset by GRPOTrainer.
    """
    rewards = []
    
    for completion, context in zip(completions, gold_context):
        # Extract all numbers used inside the steps
        # E.g., from "Step 1 : multiply(3.8, 1)", extract ["3.8", "1"]
        
        # 1. Find the text between <steps> and </steps>
        steps_match = re.search(r"<steps>(.*?)</steps>", completion, re.DOTALL)
        if not steps_match:
            rewards.append(0.0) # No steps, no reward
            continue
            
        steps_text = steps_match.group(1)
        
        # 2. Find all numbers inside the operator parentheses
        # This regex looks for numbers like 3.8, 1, 1000 inside parentheses
        # It ignores `#0` references because they start with #
        numbers_used = re.findall(r"[\(|,]\s*([\d\.]+)\s*[\)|,]", steps_text)
        
        if len(numbers_used) == 0:
            rewards.append(0.0)
            continue
            
        # 3. Check if these numbers exist in the context
        grounded_count = 0
        for num in numbers_used:
            # We check if the exact number string exists in the context text
            if str(num) in context:
                grounded_count += 1
                
        # Score is percentage of numbers that were grounded
        score = grounded_count / len(numbers_used)
        rewards.append(score)
        
    return rewards


def execute_step(operator, args, intermediate_results):
    """Helper function to execute a single DSL operation."""
    # Convert args to floats. If it's a reference like '#0', get it from intermediate_results
    parsed_args = []
    for arg in args:
        arg = arg.strip()
        if arg.startswith("#"):
            idx = int(arg[1:])
            parsed_args.append(intermediate_results[idx])
        else:
            parsed_args.append(float(arg))
            
    # Execute the math
    if operator == "add":
        return parsed_args[0] + parsed_args[1]
    elif operator == "subtract":
        return parsed_args[0] - parsed_args[1]
    elif operator == "multiply":
        return parsed_args[0] * parsed_args[1]
    elif operator == "divide":
        return parsed_args[0] / parsed_args[1]
    # Add other operators as needed...
    else:
        raise ValueError(f"Unknown operator: {operator}")

def execution_reward(completions, gold_answer, **kwargs):
    """
    Executes the model's program and compares to the gold answer.
    """
    rewards = []
    
    for completion, expected_answer in zip(completions, gold_answer):
        try:
            # 1. Extract steps text
            steps_match = re.search(r"<steps>(.*?)</steps>", completion, re.DOTALL)
            if not steps_match:
                rewards.append(0.0)
                continue
                
            steps_text = steps_match.group(1).strip().split('\n')
            
            # 2. Execute the steps
            intermediate_results = []
            for step_line in steps_text:
                # Parse "Step 1 : add(5, 3)" -> operator="add", args=["5", "3"]
                match = re.search(r"Step \d+\s*:\s*([a-zA-Z\-]+)\((.*)\)", step_line)
                if not match:
                    continue
                    
                operator = match.group(1)
                args = match.group(2).split(',')
                
                result = execute_step(operator, args, intermediate_results)
                intermediate_results.append(result)
                
            # 3. Check if final calculated result matches gold answer
            final_result = intermediate_results[-1]
            expected = float(expected_answer)
            
            # Allow 1% error margin for floating point math
            if abs(final_result - expected) / (abs(expected) + 1e-8) < 0.01:
                rewards.append(1.0) # PERFECT!
            else:
                rewards.append(0.0) # Wrong math
                
        except Exception as e:
            # If parsing fails, division by zero, etc., reward is 0
            rewards.append(0.0)
            
    return rewards


