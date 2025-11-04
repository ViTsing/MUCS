#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using a causal LM model on Self-Instruct style evaluation set")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="test/selfinstruction_eval.jsonl",
        help="Path to the Self-Instruct JSON/JSONL file"
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


def load_selfinstruction_file(path):
    """
    支持两种格式：
    1) 完整 JSON 列表： [ {..}, {..}, ... ]
    2) JSONL：每行一个 JSON 对象
    返回 Python 列表，每个元素是一个 task dict，task 中包含 "instruction" 和 "instances" 等字段
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        if not text:
            return []
        # 先尝试按完整 JSON 读取
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
                # skip invalid lines
                continue
        return items


def build_prompt(instruction: str, input_text: str):
    """
    构造 prompt。你可以根据需要替换成更复杂的模板。
    """
    if input_text and input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt


def generate_response(prompt, model, tokenizer, device, max_new_tokens, temperature, top_p) -> str:
    """Generate a response for a given prompt using the specified model and tokenizer"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    # some tokenizers/models (如 LLama) 在没有 pad_token 时需要设置 pad_token_id
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
    # 尝试去掉 prompt 前缀（若模型回显了 prompt）
    if decoded.startswith(prompt):
        generated = decoded[len(prompt):].strip()
    else:
        # 在某些 tokenizer 解码会在 prompt 前有额外空格或换行，尽量移除 prompt 内容再返回
        # 最保守的做法：如果 prompt 的最后一部分能在 decoded 中找到，就切掉到该位置之后
        idx = decoded.find(prompt.strip())
        if idx != -1:
            generated = decoded[idx + len(prompt.strip()):].strip()
        else:
            # 否则完整返回解码文本（去掉可能的前后空白）
            generated = decoded.strip()
    return generated


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 如果 tokenizer 没有 pad_token，尝试和 eos 做相同设置（避免 generate 报错）
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试以 float16 加载（若显存不足，可改为 float32）
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except Exception:
        # 回退到默认加载（更兼容但更慢/占内存）
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )

    model.to(device).eval()

    # Read evaluation set (support JSON list or JSONL)
    test_tasks = load_selfinstruction_file(args.input_file)

    results = []
    for task in tqdm(test_tasks, desc="Processing tasks"):
        # Self-Instruct 的 task 可能包含 'instruction' 和 'instances' 列表
        instruction = task.get('instruction') or task.get('Instruction') or task.get('prompt') or ''
        instances = task.get('instances') or task.get('Instances') or []
        # 如果没有 instances，但 task 自身有 input/output 直接当作一个 instance
        if not instances:
            # 有些版本里 single sample 会直接放在 task['input']/'output'
            input_text = task.get('input', '') or task.get('Input', '') or ''
            # 构造 prompt 并生成
            prompt = build_prompt(instruction, input_text)
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
                "instruction": instruction,
                "input": input_text,
                "response": response_text
            })
        else:
            # 为每个 instance 生成一次
            for inst in instances:
                input_text = inst.get('input', '') or inst.get('Input', '') or ''
                prompt = build_prompt(instruction, input_text)
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
                    "instruction": instruction,
                    "input": input_text,
                    "response": response_text
                })

    # Save outputs
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Generated {len(results)} responses and saved to {args.output_file}")


if __name__ == "__main__":
    main()
