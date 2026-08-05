import json
import pandas as pd

def expanded_table(table):
    formatted_str = ''
    for row in table:
        for cell in row:
            formatted_str+=cell+'|'
        formatted_str += '\n'
    return formatted_str

def prepare_rl_sample(sample):
    pre_text = ''.join(sample['pre_text'])
    table = sample['table']
    post_text = ''.join(sample['post_text'])
    question = sample['qa']['question']
    answer = sample['qa']['exe_ans']
    
    # Create the user prompt
    user_content = f"{pre_text} \n\n {expanded_table(table)} \n\n {post_text} \n\n {question} \n\n"
    
    # The context for grounding (everything except the question)
    gold_context = f"{pre_text} {expanded_table(table)} {post_text}"

    return {
        'messages': [{'role': 'user', 'content': user_content}],
        'gold_answer': answer,
        'gold_context': gold_context
    }

print("Loading train.json...")
df = pd.read_json("train.json")

# SFT used the first 10%. We use the remaining 90% for RL.
sft_split_idx = int(0.1 * len(df))
rl_train_data = df.iloc[sft_split_idx:].copy()

print(f"Total samples for RL: {len(rl_train_data)}")

print("Formatting RL dataset...")
rl_formatted = rl_train_data.apply(prepare_rl_sample, axis=1).tolist()

print("Saving to RL_train_formatted.json...")
with open("RL_train_formatted.json", "w") as f:
    json.dump(rl_formatted, f, indent=4)

print("Done!")
