# BoN vs PrivBoN vs ITP vs PrivITP

The main comparison experiment. Evaluates four candidate-selection algorithms
on a scored response dataset using the best `sigma` and `beta` values selected
from the upstream sweeps:

| Algorithm | Description |
|---|---|
| **BoN**     | Standard Best-of-N (argmax of proxy rewards). |
| **PrivBoN** | Gumbel-style softmax sampling with temperature `sigma`. |
| **ITP**     | Pessimistic BoN with rejection sampling, parameter `beta`. |
| **PrivITP** | Two-phase private ITP: independent `N(0, sigma_phase)` perturbations are added to (1) the threshold `lambda_hat` and (2) the rewards used in rejection sampling, with acceptance probabilities averaged over Monte Carlo realizations. |

The output is a two-panel figure (accuracy lift, expected proxy reward) plus a
printed terminal table of per-N statistics for each algorithm.

## Requirements

- Python 3.9+
- Packages: `numpy`, `matplotlib`, `tqdm`

```bash
pip install numpy matplotlib tqdm
```

## Input

A scored JSONL file where each line contains a
`responses` array, where each response has at least:

- `is_correct` (0 or 1)
- `proxy_reward` (float; may be `null` and will be skipped)

## Run

```bash
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename comparison_results \
    --sigma_gumbel 1.0 \
    --beta 0.05 \
    --n_max_exp 12 \
    --m_replicates 50 \
    --s_gauss 150 \
    --rm_label "Reward Model" \
    --dataset_label "Evaluation Set"
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input_file` | Scored JSONL from Experiment 01. | required |
| `--output_basename` | Output filename without extension. | `comparison_results` |
| `--sigma_gumbel` | Best `sigma` from the PrivBoN-Gumbel sweep (normalized space). | `1.0` |
| `--beta` | Best `beta` from the ITP sweep (normalized space). | `0.05` |
| `--sigma_phase` | Per-phase noise std for PrivITP. | `sigma_gumbel / 2` |
| `--n_max_exp` | N grid is `[2^0, ..., 2^n_max_exp]`. | `12` |
| `--m_replicates` | Monte Carlo replicates per (prompt, N). | `50` |
| `--s_gauss` | Gaussian noise samples for PrivITP per replicate. | `150` |
| `--seed` | Random seed. | `42` |
| `--rm_label` | Label shown in plot/print output. | `Reward Model` |
| `--dataset_label` | Label shown in plot/print output. | `Evaluation Set` |


## Output

- `<output_basename>.pdf`
- `<output_basename>.png`
- Stdout tables of mean accuracy lift and expected proxy reward (with standard
  errors over prompts) for each algorithm at each N.

## Reproducing across reward models / base models

To run the same comparison on a different scored dataset, just point
`--input_file` at the new JSONL and re-pass the `sigma_gumbel` / `beta`
selected from the corresponding upstream sweeps for that combination.
