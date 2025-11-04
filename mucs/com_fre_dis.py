import os
import gzip
import json
import torch
import logging
import argparse
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# 命令行示例：python com_fre_dis.py --model=qwen2.5-7b

def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ref_dat", type=str, default="c4") # 参考语料库
    parser.add_argument("--fil_num", type=int, default=15) # 文件数量
    parser.add_argument("--model", type=str, default="pythia-2.8b") # 用这个模型的分词器分个词而已
    parser.add_argument("--max_tok", type=int, default=1024)
    parser.add_argument("--sav_int", type=int, default=1e4) # 保存间隔
    arg = parser.parse_args()
    return arg


def tok_fre_dis(ref_dat, tok, fre_dis, max_tok, k):
    """
    用于计算给定文本的 token 频率分布
    """
    for i, e in enumerate(tqdm(ref_dat, desc=f"{k+1} sub-dataset")):
        text = e["text"]
        input_ids = tok.encode(text)[:max_tok] # 最多处理前 max_tok 个 token
        for input_id in input_ids:
            if input_id < len(fre_dis):
                fre_dis[input_id] += 1 # 将当前token的频率计数加1
            else:
                for i in range(input_id-len(fre_dis)+1): # token ID 超过分词器范围了
                    fre_dis.append(0)
                fre_dis[input_id] += 1


if __name__ == "__main__":
    args = get_arg()
    logging.info(f"compute token frequency distribution for {args.model} using {args.fil_num} files of {args.ref_dat}")

    # 创建输出目录
    out_dir = "output/fre_dis"
    out_path = os.path.join(out_dir, args.ref_dat)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # 加载分词器
    mod_dir = "/mnt/sdb/yyz/.cache/huggingface"
    tar_mod_name = os.path.join(mod_dir, args.model) # 目标模型的路径
    tokenizer = AutoTokenizer.from_pretrained(tar_mod_name, trust_remote_code=True)

    # 初始化频率分布数组
    fre_dis = [0] * tokenizer.vocab_size

    # 计算token频率分布
    if args.ref_dat == "c4":
        for i in range(args.fil_num):
            iter = i
            while len(str(i)) < 5:
                i = "0" + str(i)
            fil_nam = f"c4-train.{i}-of-01024.json.gz"
            ref_dat_pat = os.path.join(args.ref_dat, fil_nam) #（例如：C4/c4-train.00000-of-01024.json.gz）
            with open(ref_dat_pat, "r+", encoding="utf8") as f:
                sub_dataset = gzip.open(ref_dat_pat, "rb")
                examples = []
                for example in tqdm(sub_dataset):
                    example = json.loads(example)
                    examples.append(example)
                tok_fre_dis(examples, tokenizer, fre_dis, args.max_tok, iter)

    with open(f"{out_path}/{args.model}.pkl", "wb") as f:
        pkl.dump(fre_dis, f)

# fre_dis：[f1,f2,f3,...]
# f1：token的频数