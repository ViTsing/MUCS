import requests
import json
from tqdm import tqdm


API_KEY = ""  # 替换为你的 API Key

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://my-domain.com",  # 如果是本地开发，可以省略
    "X-Title": "Your App Name"  # 可选，用于标识你的应用
}


def generate_response(prompt):
    data = {
        "model": "openai/gpt-4-turbo",   #TODO: 裁判！  openai/gpt-4-turbo   openai/gpt-4.1-nano   deepseek/deepseek-chat-v3-0324
        "messages": [
            {"role": "system",
             "content": "你是一个指令微调领域的裁判。"},
            {"role": "user", "content":prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if 'choices' in response.json():
        return response.json()['choices'][0]['message']['content']
    else:
        return ''

#
with open('new_inputs/inputs_llama_alpaca_car_1000.json', 'r', encoding='utf-8') as f:   
    inputs_0 = json.load(f)
# 
with open('new_inputs/inputs_llama_alpaca_car_ref.json', 'r', encoding='utf-8') as f:  
    inputs_1 = json.load(f)

outputname = 'new_outputs/outputs_llama_car'

# inputs = []
for i, input in enumerate(tqdm(inputs_0[542:])):
    output = {}

    prompt = f"""[Question]
    {input['instruction']}
    [The Start of Assistant 1’s Answer]
    {inputs_0[i]['response']}
    [The End of Assistant 1’s Answer]
    [The Start of Assistant 2’s Answer]
    {inputs_1[i]['response']}
    [The End of Assistant 2’s Answer]
    [System]
    Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to  the user question displayed below. You should choose  the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should  consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their  responses. Begin your evaluation by comparing the  two responses and think a short explanation. Avoid  any positional biases and ensure that the order in which  the responses were presented does not influence your  decision. Do not allow the length of the responses to  influence your evaluation. Do not favor certain names  of the assistants. Be as objective as possible. Please don't output your explanation, just output your final verdict by strictly following this format: output “1” if Assistant 1’s Answer is better; “2” if  Assistant 2’s Answer is better; “Tie” if they are of equal quality.  \n\n**Format your output exactly like this** (without additional text):\n\nEvaluation: <1/2/Tie>\n\n
    """

    result = generate_response(prompt)
    if result == '':
        break

    output['instruction'] = input['instruction']
    output['evaluation_result'] = result   #TODO: 收集“裁判”结果

    with open(f'{outputname}_p.jsonl', 'a', encoding='utf-8') as f:   # 一行行地写入
        f.write(json.dumps(output, ensure_ascii=False) + '\n')

    # inputs.append(output)


for i, input in enumerate(tqdm(inputs_0[:])):
    output = {}

    prompt = f"""[Question]
    {input['instruction']}
    [The Start of Assistant 1’s Answer]
    {inputs_1[i]['response']}
    [The End of Assistant 1’s Answer]
    [The Start of Assistant 2’s Answer]
    {inputs_0[i]['response']}
    [The End of Assistant 2’s Answer]
    [System]
    Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to  the user question displayed below. You should choose  the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should  consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their  responses. Begin your evaluation by comparing the  two responses and think a short explanation. Avoid  any positional biases and ensure that the order in which  the responses were presented does not influence your  decision. Do not allow the length of the responses to  influence your evaluation. Do not favor certain names  of the assistants. Be as objective as possible. Please don't output your explanation, just output your final verdict by strictly following this format: output “1” if Assistant 1’s Answer is better; “2” if  Assistant 2’s Answer is better; “Tie” if they are of equal quality.  \n\n**Format your output exactly like this** (without additional text):\n\nEvaluation: <1/2/Tie>\n\n
    """

    result = generate_response(prompt)
    if result == '':
        break

    output['instruction'] = input['instruction']
    output['evaluation_result'] = result   #TODO: 收集“裁判”结果

    with open(f'{outputname}_n.jsonl', 'a', encoding='utf-8') as f:   # 一行行地写入
        f.write(json.dumps(output, ensure_ascii=False) + '\n')