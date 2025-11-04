import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses using a causal LM model on Alpaca evaluation set")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the pretrained model directory"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="test/alpaca/alpaca_eval_gpt4_baseline.json",
        help="Path to the Alpaca evaluation JSON file"
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


def generate_response(prompt, model, tokenizer, device, max_new_tokens, temperature, top_p) -> str:
    """Generate a response for a given prompt using the specified model and tokenizer"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the generated text if echoed
    # return decoded[len(prompt):] if decoded.startswith(prompt) else decoded
    return decoded[:] if decoded.startswith(prompt) else decoded   #TODO: 不切分


def main():
    args = parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Read evaluation set
    with open(args.input_file, 'r', encoding='utf-8') as f:
        test_datasets = json.load(f)

    results = []
    for item in tqdm(test_datasets, desc="Generating responses"):
        prompt = item.get('instruction', '')
        # Alpaca eval has no 'input' field
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
            'instruction': prompt,
            'input': item.get('input', ''),
            'response': response_text
        })

    # Save outputs
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
