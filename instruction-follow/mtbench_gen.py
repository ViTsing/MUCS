#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mtbench_eval.py

用法示例:
python run_mtbench_eval.py --model_path /path/to/model --input_file test/mtbench.jsonl --output_file mtbench_out.json
"""

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using a causal LM model on MT-Bench style evaluation set")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="test/mtbench_eval.jsonl",
        help="Path to the MT-Bench evaluation JSON/JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to write the generated responses JSON"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Nucleus sampling probability"
    )
    return parser.parse_args()


def load_mtbench_file(path):
    """
    支持两种格式：
    1) 完整 JSON 列表（整个文件是一个 list）
    2) JSONL（每行一个 JSON 对象）
    返回 Python 列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        # 先尝试当作完整 JSON 列表
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        # 回退到按行解析 JSONL
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # 跳过无法解析的行
                continue
        return items


def build_prompt_from_turns(turns):
    """
    给定 MT-Bench 的 turns 列表（多轮），构造一个 prompt：
    - 将 turns[:-1] 作为上下文（按序列出为 User: ...）
    - 最后一条 turn 作为当前用户输入，要求 Assistant 给出回复
    返回 prompt 字符串，以及将保存到 output JSON 的 instruction 字段（这里我们将 instruction 设为把 turns 用 ' <TURN> ' 连接的字符串）
    """
    if not turns:
        return "", ""
    # instruction 字段：保留全部 turns 的拼接（便于后续分析）
    instruction_field = " <TURN> ".join(turns)
    # prompt：把前面的用户 utterances 当作上下文（只列出 User），最后让 Assistant 回答最后一个 turn
    if len(turns) == 1:
        prompt = f"User: {turns[0]}\nAssistant:"
    else:
        ctx = ""
        for t in turns[:-1]:
            ctx += f"User: {t}\n"
        # 最后一条为当前需要回答的用户输入
        prompt = ctx + f"User: {turns[-1]}\nAssistant:"
    return prompt, instruction_field


def generate_response(prompt, model, tokenizer, device, max_new_tokens, temperature, top_p) -> str:
    """Generate a response for a given prompt using the specified model and tokenizer"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 尝试去掉 prompt 前缀（若模型输出中包含 prompt）
    cleaned = decoded
    # 若 decoded 明确以 prompt 开头，则去除 prompt 部分
    if decoded.startswith(prompt):
        cleaned = decoded[len(prompt):].strip()
    else:
        # 更保守的尝试：若能找到 prompt 的末尾子串，则切掉到该位置之后
        idx = decoded.find(prompt.strip())
        if idx != -1:
            cleaned = decoded[idx + len(prompt.strip()):].strip()
        else:
            cleaned = decoded.strip()
    return cleaned


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试以 float16 加载以节省显存
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )

    model.to(device).eval()

    # 读取 MT-Bench 数据集
    test_items = load_mtbench_file(args.input_file)

    results = []
    for item in tqdm(test_items, desc="Generating responses"):
        turns = item.get('turns') or []
        # 如果没有 turns，跳过
        if not turns:
            continue
        prompt, instruction_field = build_prompt_from_turns(turns)
        # 生成一次回复（基于整个上下文）
        response_text = generate_response(
            prompt,
            model,
            tokenizer,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p
        )
        results.append({
            "instruction": instruction_field,  # 保存全部 turns 的拼接，便于分析
            "input": "",                        # MT-Bench 不强制单独 input 字段，这里保留空字符串
            "response": response_text
        })

    # 保存输出
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Generated {len(results)} responses and saved to {args.output_file}")


if __name__ == "__main__":
    main()
