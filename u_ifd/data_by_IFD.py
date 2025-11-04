import torch
import json
import numpy as np
import argparse
from tqdm import tqdm

'''
python cherry_selection/data_by_IFD.py \
    --pt_data_path alpaca_data_cherry.pt \
    --json_data_path data/alpaca_data.json \
    --json_save_path alpaca_data_cherry.json \
    --model_name_or_path /mnt/sdb/yyz/.cache/huggingface/ifd_alpaca_pre/ \
    --max_length 512 \
    --sample_rate 0.06 \
    --prompt alpaca
'''

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')   # 微调后的模型
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--sample_number", type=int, default=0)
    parser.add_argument("--prompt", type=str, default='alpaca', help='wiz, alpaca')
    args = parser.parse_args()
    return args


import numpy as np

EPS = 1e-12  # 防止 log(0)

def compute_mean_with_rules(token_ids, loss_list, fre_dis_smo):
    """
    token_ids: 1D array-like of token ids (torch tensor or numpy array)
    loss_list: 1D array-like of per-token NLL (逐 token 的 -log p)
    fre_dis_smo: numpy array of smoothed token freqs, indexed by token id

    返回: 按你三条规则处理后的 mean 值（float）
    处理顺序说明（我采用的顺序）:
      1) 从 loss -> p: p = exp(-loss)
      2) 规则2（重复 token 使用首次出现的 p）: 把重复 token 的 p 替换为第一次出现时的 p
      3) 规则3（相邻比较阈值）: 如果 p[i] > p[i-1] 且 (p[i]-p[i-1]) >= 0.725441，则把 p[i] 设为 p[i-1]
      4) 计算值 values = -p * log(freq)，对 freq 做下界 EPS 保护
      5) 规则1（上限 0.01）: 如果 values >= 0.01 则设为 0.01
      6) 返回 mean(values)
    """
    # 转成 numpy
    if hasattr(token_ids, 'cpu'):
        token_ids = token_ids.cpu().numpy()
    else:
        token_ids = np.array(token_ids)
    loss_arr = np.array(loss_list, dtype=float)

    # 保证长度一致
    assert len(token_ids) == len(loss_arr), f"token_ids len {len(token_ids)} != loss_list len {len(loss_arr)}"

    # 0) 计算 p（逐 token 生成概率）
    p = np.exp(-loss_arr)   # shape (L,)

    # 1) 规则2：重复 token 使用第一次出现时的 p
    first_p = {}
    for idx, tid in enumerate(token_ids):
        # tid 可能是 np.int64；用 int() 做 key
        key = int(tid)
        if key not in first_p:
            first_p[key] = p[idx]
        else:
            p[idx] = first_p[key]

    # 2) 规则3：相邻比较（基于已经处理过重复 token 的 p）
    THRESH = 0.725441
    for i in range(1, len(p)):
        if p[i] > p[i-1] and (p[i] - p[i-1]) >= THRESH:
            p[i] = p[i-1]

    # 3) 计算参考频率 fre（做下界保护）
    fre = fre_dis_smo[token_ids]
    fre = np.maximum(fre, EPS)

    # 4) 计算 values = -p * log(fre)
    values = -p * np.log(fre)

    # 5) 规则1：如果某 token 的 value >= 0.01，则设为 0.01
    values = np.minimum(values, 0.01)

    # 6) 返回 mean
    return float(values.mean())


def main():

    args = parse_args()
    print(args)

    
    import pickle
    with open('llama-1-7b.pkl','rb') as f:   #FIXME: 词频文件: llama-1-7b.pkl or qwen2.5-7b.pkl
        fre_dis = pickle.load(f)              
    fre_dis_npy = np.array(fre_dis)
    fre_dis_smo = (fre_dis_npy + 1) / (fre_dis_npy.sum() + len(fre_dis_npy))

    #FIXME: llama
    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)

    # qwen
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    with open(args.json_data_path, "r") as f:
        json_data = json.load(f)

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    for i in tqdm(range(len(pt_data))):

        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        json_data_i = json_data[i]
        instruct_i = json_data_i['instruction']
        output_i = json_data_i['output']

        direct_answer_text = '### Response:' + output_i
        if args.prompt == 'wiz':
            whole_text = instruct_i+'\n\n### Response:'+output_i
        elif args.prompt == 'alpaca':
            input_i = json_data_i['input']
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

        # Tokenize the input text
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to('cpu')
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):

            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = tokenizer.encode(text_temp)
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]

            loss_list = loss_list_[start_token-1:end_token_real-1] 

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if args.max_length-instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i, args.max_length-instruct_i_len+4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, args.max_length, loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                continue

            if instruct_i_len + len_1 > args.max_length:
                continue


            #FIXME: 1 ifd
            # mean_1 = loss_list_1.mean()
            # mean_2 = loss_list_2.mean()

            # ifd + dcpdd
            # p1 = np.exp(-loss_list_1)
            # p2 = np.exp(-loss_list_2)
            # freqs1 = fre_dis_smo[token_ids_1.numpy()]
            # freqs2 = fre_dis_smo[token_ids_2.numpy()]
            # mean_1 = np.mean(-p1 * np.log(freqs1))
            # mean_2 = np.mean(-p2 * np.log(freqs2))

            # ifd + dcpdd + 3
            mean_1 = compute_mean_with_rules(token_ids_1, loss_list_1, fre_dis_smo)
            mean_2 = compute_mean_with_rules(token_ids_2, loss_list_2, fre_dis_smo)


            mean_rate = mean_2/mean_1


            #FIXME: 2 ifd            
            # if mean_rate > 1: 
            #     continue

            # ifd + dcpdd  && ifd + dcpdd +3
            if mean_rate < 1: 
                continue


            mean_rate_list.append((mean_rate,i))
            mean_list_1.append((mean_1,i))
            mean_list_2.append((mean_2,i))

        else:
            continue

    print('Do Rate')
    mean_rate_list = sorted(mean_rate_list)
    if args.sample_number == 0:
        args.sample_number = int(len(mean_rate_list)*args.sample_rate)


    #FIXME: 3 ifd
    # mean_rate_list_id = [i for i in range(len(mean_rate_list))][-args.sample_number:]
    # mean_rate_list_id_sample = [mean_rate_list[id][1] for id in mean_rate_list_id]

    # ifd + dcpdd
    mean_rate_list_id = [j for j in range(len(mean_rate_list))][:args.sample_number]
    mean_rate_list_id_sample = [mean_rate_list[j][1] for j in mean_rate_list_id]


    mean_rate_list_id_sample = sorted(mean_rate_list_id_sample)

    new_data = [json_data[idx] for idx in mean_rate_list_id_sample]
    print('New data len \n',len(new_data))
    with open(args.json_save_path, "w") as fw:
        json.dump(new_data, fw, indent=4)


if __name__ == '__main__':
    main()