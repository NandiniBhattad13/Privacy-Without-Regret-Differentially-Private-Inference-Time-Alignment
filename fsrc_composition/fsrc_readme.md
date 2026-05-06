# PrivITP Privacy Budget Sweep (FRSC)

This experiment evaluates PrivITP under a **fixed total privacy budget** across
several noise scales `sigma`. Unlike the earlier experiments, which fix N and
report metric curves, this one fixes the budget and reports how many prompts
the algorithm can process before exhausting it.

## What the script does

For each `sigma`, repeated across multiple seeds:

1. Stream prompts in a random order.
2. For each prompt, run sequential PrivITP with `n_cap` candidates per phase
   (Phase 1: noisy `lambda_hat` from a clean lambda computation; Phase 2:
   single-shot rejection sampling on noisy rewards, recording the actual
   stopping time `t_processed` of the first acceptance).
3. After each prompt, compute the realized per-prompt privacy cost using a
   numerical RDP-style accountant (`compute_eps_2_post`) plus the deterministic
   Phase-1 cost `Delta_r / sigma_X`.
4. Subtract from the running budget. Stop when the budget runs out.

For each `sigma`, the script reports (averaged over seeds):

| Quantity | Meaning |
|---|---|
| `counter`             | Number of prompts processed before budget exhaustion. |
| `avg_stopping_time`   | Mean rejection-sampling stopping time per prompt. |
| `accuracy`            | Fraction of accepted responses that were correct. |
| `T` (basic comp.)     | Basic-composition baseline = `ebudget / max(eps_per_prompt)`. |

## Requirements

- Python 3.9+
- Packages: `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

## Input

A scored dataset Both formats are accepted:

- JSONL (one record per line)
- A single JSON array of records

Each record must contain a `responses` array; each response must have:

- `proxy_reward` (float; may be `null` and will be skipped)
- `is_correct` (0 or 1)

Each prompt must have at least `--n_cap` candidate responses; prompts with
fewer are dropped.

## Run

```bash
python fsrc.py \
    --input_file /path/to/scored_responses.json \
    --output_basename privitp_sigma_sweep \
    --beta 0.05 \
    --sigmas 1.0 5.0 8.0 10.0 \
    --n_cap 16 \
    --ebudget 50.0 \
    --truncation_L 0.5 \
    --seeds 0 1 2 3 4 5 6 7 8 9
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input_file` | Scored data (JSONL or JSON array). | required |
| `--output_basename` | Output filename without extension. | `private_itp_sigma_sweep` |
| `--beta` | Pessimism strength. | `0.05` |
| `--sigmas` | Total noise budgets to sweep. Each is split equally between phases. | `1.0 5.0 8.0 10.0` |
| `--n_cap` | Candidates considered per phase per prompt. | `16` |
| `--ebudget` | Total cumulative privacy budget. | `50.0` |
| `--truncation_L` | Constant `L` in the truncation envelope. | `0.5` |
| `--seeds` | Seeds to average results over. | `0..9` |
| `--n_grid` | Grid size for the numerical expectation in the accountant. | `512` |

## Notes

- **Reward space:** rewards are globally z-score normalized at the start of
  each (sigma, seed) run. The `--sigmas` and `--beta` values are therefore in
  normalized reward units, consistent with the beta and sigma sweep scripts.
- **Sequential rejection sampling:** Phase 2 runs an actual Bernoulli draw at
  each candidate and stops at the first acceptance. The realized stopping
  time is what feeds into the accountant; this is why the realized epsilon
  spend is typically lower than the worst-case `eps_max_P`.
- **Basic composition baseline:** `T = ebudget / max_p eps_max_p` represents
  what a non-adaptive accountant (one that always assumes the worst-case
  per-prompt epsilon) would allow. The `counter` bar exceeding `T` is the
  headline takeaway of the experiment.
- **Accountant:** `compute_eps_2_post` uses `scipy.stats.norm.logcdf` over a
  uniform grid on `[0, 1]` of size `--n_grid` to numerically estimate the
  log-ratio bound; raise `--n_grid` if you need tighter estimates.

## Output

- `<output_basename>.png` and `<output_basename>.pdf` (grouped bar chart with
  per-sigma `# prompts processed`, `avg stopping time`, `avg accuracy`, plus a
  red dashed segment indicating the basic-composition baseline `T`).
- A summary line per `sigma` printed to stdout with means and standard
  deviations across seeds.
