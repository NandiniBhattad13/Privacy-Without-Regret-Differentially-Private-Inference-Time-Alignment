# ITP vs PrivITP 

This experiment compares non-private ITP against PrivITP across multiple values
of `sigma`, but with one important twist: each
prompt's candidate pool is **split into two disjoint halves** of size N. The
first half is used to compute the threshold `lambda_hat`; the second half is
used as the rejection-sampling pool. This data split keeps phase 1 and phase 2
statistically independent.

## What gets compared

| Algorithm | Description |
|---|---|
| **ITP**     | Standard pessimistic BoN. Uses only the phase-2 pool with a clean `lambda_hat` computed on phase 2. |
| **PrivITP** | Two-phase private ITP. `lambda_hat` is computed on the phase-1 pool with `N(0, sigma_X)` noise added; the phase-2 rewards used in rejection sampling have `N(0, sigma_Z)` noise added. By default `sigma_X = sigma_Z = sigma / 2`. |

The truncation envelope used in PrivITP is `R_max(phase2) + sigma_Z * L`, where
`L` is configurable.

## Requirements

- Python 3.9+
- Packages: `numpy`, `matplotlib`, `tqdm`

```bash
pip install numpy matplotlib tqdm
```

## Input

A scored JSONL file produced by Experiment 01. Because of the two-phase split,
**each prompt must have at least `2 * 2^n_max_exp` candidate responses**.

Each response must contain:

- `is_correct` (0 or 1)
- `proxy_reward` (float; may be `null` and will be skipped)

## Run

```bash
python run.py \
    --input_file ../01-dataset-generation/scored_responses.jsonl \
    --output_basename itp_vs_privitp_split \
    --beta 0.1 \
    --sigmas 0.5 1.5 4.0 \
    --n_max_exp 12 \
    --m_replicates 50 \
    --s_gauss 150 \
    --truncation_L 4.0
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input_file` | Scored JSONL from Experiment 01. | required |
| `--output_basename` | Output filename without extension. | `itp_vs_privitp_split` |
| `--beta` | Pessimism strength (raw reward space; see note below). | `0.1` |
| `--sigmas` | Total noise budgets to sweep over. Each is split equally between phases. | `0.5 1.5 4.0` |
| `--n_max_exp` | N grid is `[2^0, ..., 2^n_max_exp]`. Needs `2 * 2^n_max_exp` candidates per prompt. | `12` |
| `--m_replicates` | Monte Carlo replicates per (prompt, N). | `50` |
| `--s_gauss` | Number of Gaussian noise samples per PrivITP call. | `150` |
| `--truncation_L` | Constant `L` in the truncation envelope. | `4.0` |
| `--seed` | Random seed. | `42` |
| `--rm_label` | Label for plot/print output. | `Reward Model` |
| `--dataset_label` | Label for plot/print output. | `Evaluation Set` |

## Important notes

- **Reward space:** this script operates on **raw proxy rewards**. The
  `--sigmas` and `--beta` values are therefore in raw reward units. If
  reusing best hyperparameters, multiply the normalized values by the
  raw reward standard deviation to get the equivalent raw-space values.
- **Sample budget:** the data-split design halves the effective N per phase, so
  you need twice as many candidate responses per prompt. With the default
  `--n_max_exp 12`, every prompt must have at least 8192 candidates.
- The ITP baseline reported in the plot uses only the phase-2 pool, so its
  performance can differ slightly from the ITP curve in the comparison of
  all four algorithms (which uses a single shared pool of size N).

## Output

- `<output_basename>.pdf`
- `<output_basename>.png`
- Per-N statistics (mean lift +/- std error) printed to stdout for the ITP
  baseline and each PrivITP sigma variant.
