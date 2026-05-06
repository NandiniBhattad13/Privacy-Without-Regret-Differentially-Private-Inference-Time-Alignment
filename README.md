# Privacy-Without-Regret-Differentially-Private-Inference-Time-Alignment

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The repository was developed and tested on Python 3.9+. Response generation and reward-model scoring requires a CUDA-capable GPU. All other experiments can be run on CPU using only `numpy`, `scipy`, `matplotlib`, and `tqdm`.

If the base policy or reward model is gated on the Hugging Face Hub, set:

```bash
export HF_TOKEN=<your_huggingface_token>
```

The base policies and reward models used in the paper are public, pretrained models loaded directly from the Hugging Face Hub (see *Pre-trained Models* below).

## Evaluation

Given a scored dataset for one (base model, reward model) pair, the evaluation pipeline has four stages:

1. **Hyperparameter sweep over `beta`** for ITP (Experiment 02).
2. **Hyperparameter sweep over `sigma`** for PrivBoN (Experiment 03).
3. **Main four-way comparison** (BoN, PrivBoN, ITP, PrivITP) at the best `beta` and `sigma` from stages 1–2, plus the **ITP-vs-PrivITP experiment** at varying sigma values.
4. **Fixed-budget rejection-sampling controller (FRSC)** experiment under a fixed total privacy budget.

### Step 1: select beta (ITP sweep)

```eval-beta
cd experiments/02-pessimistic-bon-beta-sweep
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename beta_sweep \
    --betas 0.0005 0.005 0.01 0.05 0.1 0.5 1.0 \
    --rm_label "<reward_model_name>" \
    --dataset_label "<dataset_name>"
```

Inspect the resulting plot and pick the `beta` that maximizes accuracy lift at the largest N. Call this `BETA_STAR`.

### Step 2: select sigma (PrivBoN sweep)

```eval-sigma
cd ../03-private-bon-sigma-sweep
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename sigma_sweep_accuracy \
    --metric accuracy_lift \
    --sigmas 0.5 0.75 1.0 1.25 1.5 2.0
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename sigma_sweep_reward \
    --metric estimated_reward \
    --sigmas 0.1 0.2 0.5 1.0 1.5 2.0
```

The accuracy-lift plot is the primary selection signal; the estimated-reward plot serves as a complementary diagnostic. Call the chosen value `SIGMA_STAR`.

### Step 3: main comparison and data-split experiment

Plug `BETA_STAR` and `SIGMA_STAR` into the comparison experiment:

```eval-comparison
cd ../04-bon-vs-privbon-vs-itp-vs-privitp
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename comparison_results \
    --sigma_gumbel <SIGMA_STAR> \
    --beta <BETA_STAR>
```

Run the ITP-vs-PrivITP data-split experiment on the same dataset (this experiment requires `2 * 2^n_max_exp` candidate responses per prompt because of the disjoint phase split; see its README for details):

```eval-split
cd ../05-itp-vs-privitp-data-split
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename itp_vs_privitp_split \
    --beta <BETA_STAR>
```

### Step 4: privacy budget experiment (FRSC)

```eval-frsc
cd ../06-privitp-privacy-budget-sweep
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename privitp_budget \
    --beta <BETA_STAR> \
    --sigmas 1.0 5.0 8.0 10.0 \
    --ebudget 50.0
```

Each evaluation script saves a PDF and a PNG. Experiments 04, 05, and 06 additionally print per-N (or per-sigma) statistics to stdout in a format suitable for assembling LaTeX tables.

## Pre-trained Models

This repository does not train or release any models. The base policies and reward models used in the paper are public, pretrained models on the Hugging Face Hub and are loaded directly by Experiment 01 via the `--model_name` and `--reward_model_name` arguments.
