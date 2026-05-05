"""
Generate N candidate responses per MMLU prompt using a base policy model via vLLM.

Outputs a JSONL file where each line contains a prompt, its choices, ground-truth
answer, subject, and a list of N sampled responses.
"""

import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses on MMLU subjects with vLLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model identifier on Hugging Face Hub.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="generated_responses.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["college_mathematics", "college_chemistry"],
        help="MMLU subjects to include.",
    )
    parser.add_argument("--n_samples", type=int, default=10000, help="Responses per prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="CUDA device id(s) to expose to this process (e.g. '0' or '0,1').",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
    )
    return parser.parse_args()


def format_prompt(tokenizer, question, choices):
    options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    content = (
        "Solve the following multiple-choice question step by step. "
        "End your response with 'The answer is [A/B/C/D]'.\n\n"
        f"Question: {question}\n{options_text}"
    )
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "Let me think step by step.\n"


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    from vllm import LLM, SamplingParams

    target_subjects = set(args.subjects)

    print("Loading and filtering MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.filter(lambda x: x["subject"] in target_subjects)

    questions = dataset["question"]
    choices_list = dataset["choices"]
    subjects = dataset["subject"]
    gt_answers = [["A", "B", "C", "D"][ans] for ans in dataset["answer"]]

    print(f"Loaded {len(questions)} prompts across subjects: {sorted(target_subjects)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Booting vLLM engine...")
    llm = LLM(
        model=args.model_name,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop=["Question:", "\n\nLet's", "\n\n**", "User:", "<end_of_turn>"],
    )

    # Resume support
    processed_prompts = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_prompts.add(data["prompt"])
                except json.JSONDecodeError:
                    continue
        print(f"[RESUME] Found existing file. Skipping {len(processed_prompts)} prompts.")

    with open(args.output_file, "a") as f:
        iterator = zip(questions, choices_list, gt_answers, subjects)
        for q, choices, ans, subject in tqdm(
            iterator, total=len(questions), desc="Generating"
        ):
            if q in processed_prompts:
                continue

            formatted_input = format_prompt(tokenizer, q, choices)
            outputs = llm.generate([formatted_input], sampling_params, use_tqdm=False)
            responses = [{"text": output.text} for output in outputs[0].outputs]

            record = {
                "prompt": q,
                "choices": choices,
                "subject": subject,
                "gt_answer": ans,
                "responses": responses,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"\nGeneration complete. Data saved to {args.output_file}.")


if __name__ == "__main__":
    main()
