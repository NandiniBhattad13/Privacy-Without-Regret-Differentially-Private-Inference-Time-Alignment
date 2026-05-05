"""
Score generated responses with a sequence-classification reward model.

Reads the JSONL produced by `generate_responses.py`, applies a reward model to
each (prompt, response) pair, extracts the predicted MMLU letter answer from
each response, and writes an enriched JSONL with `predicted_answer`,
`is_correct`, and `proxy_reward` fields per response.
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Score generated responses with a reward model.")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL from generation step.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL with scores.")
    parser.add_argument(
        "--reward_model_name",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
        help="Reward model identifier on Hugging Face Hub.",
    )
    parser.add_argument(
        "--rm_format",
        type=str,
        default="oasst",
        choices=["oasst", "plain"],
        help="Prompt formatting style for the reward model.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA device id (e.g. '0').")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def extract_mmlu_answer(text):
    """Extract an A/B/C/D answer from a model's response."""
    match = re.search(r"The answer is\s*[:\s]*\**([A-D])\**", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
    return None


def format_rm_input(question, choices, response, style="oasst"):
    options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    full_prompt = f"Question: {question}\n{options_text}\nLet me think step by step."

    if style == "oasst":
        return f"<|prompter|>{full_prompt}<|endoftext|><|assistant|>{response}<|endoftext|>"
    return f"{full_prompt}\n\n{response}"


def get_torch_dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading reward model: {args.reward_model_name}")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
        torch_dtype=get_torch_dtype(args.dtype),
    ).to(device)
    rm_model.eval()

    print(f"Loading generated data from {args.input_file}...")
    with open(args.input_file, "r") as f:
        all_data = [json.loads(line) for line in f]

    scored_prompts = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            for line in f:
                try:
                    scored_prompts.add(json.loads(line)["prompt"])
                except Exception:
                    continue
        print(f"[RESUME] Skipping {len(scored_prompts)} already-scored prompts.")

    with open(args.output_file, "a") as f:
        for item in tqdm(all_data, desc="Scoring"):
            prompt = item["prompt"]
            if prompt in scored_prompts:
                continue

            choices = item["choices"]
            gt = item["gt_answer"]
            responses = item["responses"]

            for i in range(0, len(responses), args.batch_size):
                batch_responses = responses[i : i + args.batch_size]
                batch_texts = [r["text"] for r in batch_responses]

                preds = [extract_mmlu_answer(t) for t in batch_texts]
                is_correct = [
                    1 if (p == gt) and (p is not None) else 0 for p in preds
                ]

                rm_inputs_str = [
                    format_rm_input(prompt, choices, t, style=args.rm_format)
                    for t in batch_texts
                ]
                inputs = rm_tokenizer(
                    rm_inputs_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = rm_model(**inputs)
                    scores = outputs.logits.squeeze(-1).tolist()
                    if not isinstance(scores, list):
                        scores = [scores]

                for j, r in enumerate(batch_responses):
                    r["predicted_answer"] = preds[j]
                    r["is_correct"] = is_correct[j]
                    r["proxy_reward"] = scores[j]

                del inputs, outputs

            f.write(json.dumps(item) + "\n")
            f.flush()
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\nScoring complete.")


if __name__ == "__main__":
    main()
