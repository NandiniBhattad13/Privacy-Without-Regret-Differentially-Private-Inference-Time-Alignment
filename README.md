# Privacy-Without-Regret: Differentially Private Inference-Time Alignment

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The repository was developed and tested on Python 3.9+. All experiments run on CPU using only `numpy`, `scipy`, `matplotlib`, and `tqdm`.

## Input data

Each evaluation script expects a precomputed scored dataset passed via `--input_file`. The file must be either a JSONL (one record per line) or a JSON array. Each record must follow this schema:

```json
{
  "prompt": "<the prompt text>",
  "gt_answer": "<ground-truth answer>",
  "responses": [
    {
      "text": "<sampled response text>",
      "is_correct": 0,
      "proxy_reward": 1.234
    }
  ]
}
```

Each response must contain at least `is_correct` (0 or 1) and `proxy_reward` (float; entries with `null` are skipped). The number of responses per prompt must be at least as large as the maximum N used by the experiment. The ITP vs PrivITP script (`itp_vs_privitp_diff_sigma.py`) requires `2 * 2^n_max_exp` responses per prompt due to its disjoint phase split.

The paper reports results on three datasets evaluated under multiple (base policy, reward model) combinations. To reproduce a particular configuration, supply the corresponding scored file to `--input_file` for each script below.

## Evaluation

Given a scored dataset for one (base model, reward model) pair, the evaluation pipeline has four stages. In each command below, replace `<scored_data>` with the path to your scored file.

### Step 1: Select beta — ITP hyperparameter sweep

```eval-beta
python beta.py \
    --input_file <scored_data> \
    --output_basename beta_sweep \
    --betas 0.0005 0.005 0.01 0.05 0.1 0.5 1.0 \
    --rm_label "<reward_model_name>" \
    --dataset_label "<dataset_name>"
```

Inspect the output plot and pick the `beta` that maximizes accuracy lift at the largest N. Call this `BETA_STAR`.

### Step 2: Select sigma — PrivBoN hyperparameter sweep

```eval-sigma
python sigma.py \
    --input_file <scored_data> \
    --output_basename sigma_sweep_accuracy \
    --metric accuracy_lift \
    --sigmas 0.5 0.75 1.0 1.25 1.5 2.0

python sigma.py \
    --input_file <scored_data> \
    --output_basename sigma_sweep_reward \
    --metric estimated_reward \
    --sigmas 0.1 0.2 0.5 1.0 1.5 2.0
```

The accuracy-lift plot is the primary selection signal; the estimated-reward plot serves as a complementary diagnostic. Call the chosen value `SIGMA_STAR`.

### Step 3: Main four-way comparison and ITP vs PrivITP experiment

Plug `BETA_STAR` and `SIGMA_STAR` into the four-way comparison (BoN, PrivBoN, ITP, PrivITP):

```eval-comparison
python Comparison_of_ITP_BON_PrivITP_PrivBON.py \
    --input_file <scored_data> \
    --output_basename comparison_results \
    --sigma_gumbel <SIGMA_STAR> \
    --beta <BETA_STAR>
```

Run the ITP vs PrivITP experiment at varying sigma values on the same dataset. This script requires `2 * 2^n_max_exp` candidate responses per prompt due to the disjoint phase split (see `itp_comparison.md` for details):

```eval-split
python itp_vs_privitp_diff_sigma.py \
    --input_file <scored_data> \
    --output_basename itp_vs_privitp \
    --beta <BETA_STAR>
```

### Step 4: Privacy budget experiment (FRSC)

```eval-frsc
python fsrc.py \
    --input_file <scored_data> \
    --output_basename frsc_results \
    --beta <BETA_STAR> \
    --sigmas 1.0 5.0 8.0 10.0 \
    --ebudget 50.0
```

Each script saves a PDF and PNG of the output figure. The comparison and FRSC scripts additionally print per-N (or per-sigma) statistics to stdout suitable for assembling LaTeX tables.
