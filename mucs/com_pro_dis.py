import os
import json
import torch
import logging
import argparse
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger().setLevel(logging.INFO)


# 命令行示例：python com_pro_dis.py --data=qqp --tar_mod=qwen2.5-7b --key_nam=text

def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tar_mod", type=str, default="pythia-2.8b") # 目标模型
    parser.add_argument("--data", type=str, default="WikiMIA_128") # 被测数据集（得是.jsonl文件）
    parser.add_argument("--max_tok", type=int, default=1024)
    parser.add_argument("--key_nam", type=str, default="input") # 关键字
    arg = parser.parse_args()
    return arg


def load_model(tar_mod_nam):
    '''
    加载目标模型
    '''
    tar_mod = AutoModelForCausalLM.from_pretrained(tar_mod_nam, return_dict=True, trust_remote_code=True, torch_dtype=torch.float16).to('cuda')
    tar_mod.eval()
    tar_tok = AutoTokenizer.from_pretrained(tar_mod_nam, trust_remote_code=True)
    return tar_mod, tar_tok


def cal_ppl(text, model, tok):
    '''
    计算给定文本的token级概率分布，并进一步生成用于预训练数据检测的统计量（如对数概率、均值、方差）
    '''
    device = model.device
    input_ids = tok.encode(text, max_length=args.max_tok, truncation=True) # 文本编码为token ID
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_ids = input_ids.to(device) # 将输入数据移动到模型所在设备

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)

    logit = output[1]

    prob_weight = torch.nn.functional.softmax(logit, dim=-1)[0][:-1]
    prob = torch.nn.functional.log_softmax(logit, dim=-1)[0][:-1]

    prob_mu = (prob_weight * prob).sum(-1)
    prob_sigma = (prob_weight * torch.square(prob)).sum(-1) - torch.square(prob_mu)
    input_ids = input_ids[0][1:]

    probs = prob[torch.arange(len(prob)).to(device), input_ids].tolist()
    input_ids = input_ids.tolist()
    mu = prob_mu.tolist()
    sigma = prob_sigma.tolist()

    return probs, input_ids, mu, sigma


def inference(text, idx, tar_mod, tar_tok):
    '''
    对输入文本进行推理，计算目标模型和参考模型在文本上的概率分布
    '''
    response = {}
    tar_prob, input_ids, mu, sigma = cal_ppl(text, tar_mod, tar_tok) # 计算目标模型的概率分布
    low_prob, _, _, _ = cal_ppl(text.lower(), tar_mod, tar_tok) # 计算目标模型在小写文本上的概率分布

    # 填充字典
    response["input_ids"] = input_ids
    response["tar_prob"] = tar_prob
    response["low_prob"] = low_prob
    response["tar_prob_mu"] = mu
    response["tar_prob_sigma"] = sigma
    response["text"] = text
    response["idx"] = idx

    return response


def tok_pro_dis(dat, key_nam, tar_mod, tar_tok):
    '''
    批量处理数据集中的每个样本
    '''
    responses = []
    for example in tqdm(dat):
        text = example[key_nam] # 由args.key_nam指定，默认是"text"
        idx = example['idx']
        responses.append(inference(text, idx, tar_mod, tar_tok))

    return  responses


if __name__ == '__main__':
    args = get_arg()
    logging.info(f"compute token probability distribution from {args.tar_mod} on {args.data}")

    # 创建输出目录
    out_dir = "output/pro_dis"
    out_path = os.path.join(out_dir, args.data) # 构建目录（例如：output/pro_dis/WikiMIA_128）
    Path(out_path).mkdir(parents=True, exist_ok=True) # 创建目录（如果目录已存在则忽略）

    # 加载数据集
    dat_dir = "pre_detect/"
    dat_pat = os.path.join(dat_dir, f"{args.data}.jsonl") # 构建文件路径（例如：data/WikiMIA_128.jsonl）
    with open(dat_pat, 'r') as f: # 逐行读取JSONL文件，并将每行解析为Python字典
        dataset = [json.loads(line) for line in f] # 存储所有样本的列表（每个样本是一个字典）

    # 加载模型
    mod_dir = "/mnt/sdb/yyz/.cache/huggingface"
    tar_mod_name = os.path.join(mod_dir, args.tar_mod) # 目标模型的路径
    tar_model, tar_tokenizer = load_model(tar_mod_name)

    # 批量推理
    pro_dis = tok_pro_dis(dataset, args.key_nam, tar_model, tar_tokenizer)

    # 保存结果
    # 将推理结果列表序列化为.pkl文件（如output/pro_dis/WikiMIA_128/pythia-2.8B.pkl）
    with open(f"{out_path}/{args.tar_mod}.pkl", "wb") as f:
        pkl.dump(pro_dis, f)


# pro_dis：[p1,p2,p3,...,]
# p1 = {'input_ids':[],
#       'tar_prob':[],
#       'low_prob':[],
#       'tar_prob_mu':[],
#       'tar_prob_sigma':[],
#       'text':....,
#       'idx':...,
#       }
#