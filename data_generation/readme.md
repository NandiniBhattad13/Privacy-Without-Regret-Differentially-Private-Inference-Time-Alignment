# Experiment 01: Dataset Generation and Scoring

This stage produces the core dataset that all downstream experiments depend on.
It works in two steps:

1. **Generation:** sample N candidate responses per MMLU prompt from a base
   policy model using vLLM.
2. **Scoring:** score each (prompt, response) pair with a reward model and
   compute correctness against the ground-truth MMLU letter.

The output is a JSONL file where each line corresponds to one prompt and
contains its full list of scored responses.

## Requirements

- Python 3.9+
- A CUDA-capable GPU
- Packages:
  - `vllm`
  - `transformers`
  - `torch`
  - `datasets`
  - `huggingface_hub`
  - `tqdm`

Install:

```bash
pip install vllm transformers torch datasets huggingface_hub tqdm
```

If the policy or reward model is gated on the Hugging Face Hub, set:

```bash
export HF_TOKEN=<your_hf_token>
```

## Step 1: Generate responses

```bash
python generate_responses.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --subjects college_mathematics college_chemistry \
    --n_samples 10000 \
    --max_new_tokens 1024 \
    --output_file generated_responses.jsonl \
    --gpu_id 0
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--model_name` | HF model id of the base policy. | `microsoft/Phi-3-mini-4k-instruct` |
| `--subjects` | One or more MMLU subject names to filter to. | `college_mathematics college_chemistry` |
| `--n_samples` | Number of responses sampled per prompt. | `10000` |
| `--temperature`, `--top_p`, `--top_k` | Sampling parameters. | `1.0`, `0.95`, `50` |
| `--max_new_tokens` | Max generation length. | `1024` |
| `--gpu_memory_utilization` | vLLM KV-cache fraction. | `0.7` |
| `--gpu_id` | Value passed to `CUDA_VISIBLE_DEVICES`. | `0` |
| `--dtype` | `bfloat16` / `float16` / `float32` / `auto`. | `bfloat16` |

The script appends to `--output_file` and skips prompts that are already
present, so it can be resumed safely.

### Output format

Each line of `generated_responses.jsonl` looks like:

```json
{
  "prompt": "....",
  "choices": ["A text", "B text", "C text", "D text"],
  "subject": "college_mathematics",
  "gt_answer": "C",
  "responses": [
    {"text": "..."},
    {"text": "..."}
  ]
}
```

## Step 2: Score responses with a reward model

```bash
python score_responses.py \
    --input_file generated_responses.jsonl \
    --output_file scored_responses.jsonl \
    --reward_model_name OpenAssistant/reward-model-deberta-v3-large-v2 \
    --rm_format oasst \
    --batch_size 64 \
    --gpu_id 0
```

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--input_file` | JSONL produced in Step 1. | required |
| `--output_file` | Where to write scored data. | required |
| `--reward_model_name` | HF model id of the reward model. | `OpenAssistant/reward-model-deberta-v3-large-v2` |
| `--rm_format` | `oasst` (uses `<\|prompter\|>` / `<\|assistant\|>` tags) or `plain`. | `oasst` |
| `--batch_size` | Reward-model batch size. | `64` |
| `--max_length` | Tokenizer truncation length. | `1024` |
| `--gpu_id` | Value passed to `CUDA_VISIBLE_DEVICES`. | `0` |
| `--dtype` | Reward-model precision. | `float16` |

The script also supports resuming: it skips prompts that already appear in the
output file.

### Output format

Each response object in the output gains three new fields:

```json
{
  "text": "...",
  "predicted_answer": "B",
  "is_correct": 0,
  "proxy_reward": 1.832
}
```

- `predicted_answer`: letter extracted from the response, or `null`.
- `is_correct`: `1` if `predicted_answer == gt_answer`, else `0`.
- `proxy_reward`: scalar reward from the reward model.

## Notes on swapping models

To reproduce the experiments with a different base policy or reward model,
change `--model_name` and `--reward_model_name` accordingly. The scoring
script's `--rm_format` flag controls the prompt template; OpenAssistant-style
DeBERTa rewards expect the `oasst` format, while most other classifier-based
rewards work with `plain`.
