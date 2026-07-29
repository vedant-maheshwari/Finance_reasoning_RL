import json
import pandas as pd

def expanded_table(table):
    formatted_str = ''
    for row in table:
        for cell in row:
            formatted_str+=cell+'|'
        formatted_str += '\n'
    return formatted_str

def format_program(program):
    program = program.replace('), ', ')|')
    return ('\n'.join(f'Step {i+1} : {item}' for i,item in enumerate(program.split('|'))))

def prompt_generator(sample):
    pre_text = ''.join(sample['pre_text'])
    table = sample['table']
    post_text = ''.join(sample['post_text'])
    question = sample['qa']['question']
    answer = sample['qa']['exe_ans']
    program = sample['qa']['program']

    user_content = f'{pre_text} \n\n {expanded_table(table)} \n\n {post_text} \n\n {question} \n\n'
    assistant_content = f"<steps>\n{format_program(program)}</steps>\n\n<Answer>\n{answer}</Answer>"
    return {'messages':[{'role':'user', 'content': user_content}, 
            {'role':'assistant', 'content':assistant_content}]}

df = pd.read_json("Dataset/train.json")
SFT_train_data = df.copy()
SFT_train_data = SFT_train_data.iloc[0:int(0.1*len(SFT_train_data))]
SFT_train_data['messages'] = SFT_train_data.apply(prompt_generator, axis=1)

with open("SFT_train_formatted.json", "w") as f:
    json.dump(SFT_train_data['messages'].tolist(), f, indent=4)
