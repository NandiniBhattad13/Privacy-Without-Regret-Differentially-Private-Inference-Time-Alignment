"""
PrivITP under a fixed total privacy budget, swept over sigma.

For each sigma, this script processes prompts one at a time using the two-phase
PrivITP algorithm with sequential single-shot rejection sampling. After each
prompt, the realized privacy cost is computed using the per-prompt RDP-style
accountant `compute_eps_2_post` (which uses the actual stopping time and the
realized noisy threshold lambda_tilde) plus the deterministic Phase-1 cost
Delta_r / sigma_X. The loop stops when the cumulative budget is exhausted.

For each sigma we report (averaged over multiple seeds):

    counter             - number of prompts processed before the budget ran out
    avg_stopping_time   - average rejection-sampling stopping time per prompt
    accuracy            - fraction of prompts where the accepted response is correct
    T (basic comp.)     - basic-composition baseline = ebudget / max(eps_per_prompt)

The output is a grouped bar chart with three bars per sigma plus a red dashed
horizontal segment indicating T for each sigma.
"""

import argparse
import json
import os
import warnings

import matplotlib as mpl
from plot_utils import configure_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")




def parse_args():
    parser = argparse.ArgumentParser(
        description="PrivITP privacy-budget sweep over sigma."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Scored data file. Accepts either JSONL "
                             "(one record per line) or a JSON array.")
    parser.add_argument("--output_basename", type=str, default="private_itp_sigma_sweep",
                        help="Output filename without extension.")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Pessimism strength.")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[1.0, 5.0, 8.0, 10.0],
                        help="Total noise budgets to sweep over. Each is split equally "
                             "between Phase 1 (lambda) and Phase 2 (rewards).")
    parser.add_argument("--n_cap", type=int, default=16,
                        help="Maximum candidates considered per phase per prompt.")
    parser.add_argument("--ebudget", type=float, default=50.0,
                        help="Total privacy budget (cumulative epsilon).")
    parser.add_argument("--truncation_L", type=float, default=0.5,
                        help="Truncation constant L used in M = (R_max + sigma_Z*L - lambda)/beta.")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)),
                        help="Seeds to average over.")
    parser.add_argument("--n_grid", type=int, default=512,
                        help="Grid size for the numerical expectation in the accountant.")
    return parser.parse_args()


# ==========================================
# Algorithm primitives
# ==========================================
def compute_norm_constant(proxy_rewards, beta):
    N = len(proxy_rewards)
    sorted_rewards = np.sort(proxy_rewards)
    r_prev = -np.inf
    J = np.sum(sorted_rewards) / N
    Z = 1.0
    lam = (J - beta) / Z
    for i in range(N):
        lam = (J - beta) / Z
        r_curr = sorted_rewards[i]
        if (r_prev <= lam < r_curr) or (i == N - 1):
            return lam
        J -= r_curr / N
        Z -= 1.0 / N
        r_prev = r_curr
    return lam


def _log_expected_phi(arg_at_grid):
    """Stable log of the average of standard-normal CDFs over a grid of arguments."""
    log_phi = norm.logcdf(arg_at_grid)
    m = np.max(log_phi)
    return m + np.log(np.mean(np.exp(log_phi - m)))


def compute_eps_2_post(lam, R_max, Delta_r, beta, sigma, t, n_grid=512):
    """
    Per-prompt Phase-2 RDP accountant for PrivITP.

    Returns the log-ratio bound on epsilon_2 given the realized stopping time t,
    the noisy threshold lam, the prompt-specific R_max and Delta_r, and the
    Phase-2 noise scale sigma.
    """
    M = (R_max - lam) / beta
    u = (np.arange(n_grid) + 0.5) / n_grid
    bMu = beta * M * u

    arg_num_1 = (lam + bMu - R_max + Delta_r) / sigma
    arg_den_1 = (lam + bMu - R_max) / sigma
    log_ratio_1 = _log_expected_phi(arg_num_1) - _log_expected_phi(arg_den_1)

    arg_num_2 = (Delta_r - lam - bMu) / sigma
    arg_den_2 = (-lam - bMu) / sigma
    log_ratio_2 = _log_expected_phi(arg_num_2) - _log_expected_phi(arg_den_2)

    return (t - 1) * log_ratio_1 + log_ratio_2


def run_priv_itp(prompt_rewards, prompt_correct, beta, sigma,
                 sigma_X, sigma_Z, L_trunc, n_cap, rng):
    """Run a single sequential PrivITP pass on one prompt and return (t, lam_tilde, is_correct)."""
    R_max_full = float(np.max(prompt_rewards))

    # Phase 1: compute lambda on a clean subset, then perturb by N(0, sigma_X)
    idx_phase1 = rng.choice(len(prompt_rewards), size=n_cap, replace=False)
    r_phase1 = prompt_rewards[idx_phase1]
    lam = compute_norm_constant(r_phase1, beta)
    lam_tilde = lam + rng.normal(0.0, sigma_X)

    M = (R_max_full + sigma_Z * L_trunc - lam_tilde) / beta

    # Phase 2: noisy rejection sampling on a fresh subset
    idx_phase2 = rng.choice(len(prompt_rewards), size=n_cap, replace=False)
    r_phase2_clean = prompt_rewards[idx_phase2]
    r_phase2_noisy = r_phase2_clean + rng.normal(0.0, sigma, size=n_cap)
    correct_phase2 = prompt_correct[idx_phase2]

    t_processed = n_cap
    accepted_idx = n_cap - 1
    for i in range(n_cap):
        w_i = max((r_phase2_noisy[i] - lam_tilde) / beta, 0.0)
        p_i = min(w_i / M, 1.0) if M > 0 else 0.0
        if rng.random() < p_i:
            t_processed = i + 1
            accepted_idx = i
            break

    is_correct = int(correct_phase2[accepted_idx])
    return t_processed, lam_tilde, is_correct


# ==========================================
# Data loading
# ==========================================
def load_scored_data(path):
    """Load either a JSONL file or a JSON array of records."""
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


# ==========================================
# Main loop for one (sigma, seed)
# ==========================================
def run_privacy_budget_loop(all_data, beta, sigma, sigma_X, sigma_Z, L_trunc,
                            n_cap, ebudget_init, seed, n_grid):
    rng = np.random.default_rng(seed)

    # Filter prompts with enough candidates and stash (rewards, correct) pairs
    prompts_data = []
    for item in all_data:
        rewards, correct = [], []
        for r in item.get("responses", []):
            if r.get("proxy_reward") is not None:
                rewards.append(r["proxy_reward"])
                correct.append(r.get("is_correct", 0))
        if len(rewards) >= n_cap:
            prompts_data.append((np.asarray(rewards, dtype=np.float64),
                                 np.asarray(correct, dtype=np.int32)))

    # Global z-score normalization
    all_rewards_concat = np.concatenate([rw for rw, _ in prompts_data])
    global_mean = float(np.mean(all_rewards_concat))
    global_std = float(np.std(all_rewards_concat))
    if global_std == 0.0:
        global_std = 1.0
    prompts_data = [((rw - global_mean) / global_std, cc)
                    for rw, cc in prompts_data]

    order = rng.permutation(len(prompts_data))
    ebudget = ebudget_init
    counter = 0
    eps_max_array = []
    stopping_time_array = []
    correct_array = []

    for prompt_idx in order:
        rewards_P, correct_P = prompts_data[prompt_idx]
        R_max_P = float(np.max(rewards_P))
        Delta_r_P = R_max_P - float(np.min(rewards_P))

        # Worst-case accountant for this prompt (uses deterministic lambda)
        idx_emax = rng.choice(len(rewards_P), size=n_cap, replace=False)
        lam_det = compute_norm_constant(rewards_P[idx_emax], beta)
        eps_2_post_emax = compute_eps_2_post(
            lam=lam_det, R_max=R_max_P, Delta_r=Delta_r_P,
            beta=beta, sigma=sigma, t=n_cap, n_grid=n_grid,
        )
        eps_max_P = eps_2_post_emax + Delta_r_P / sigma_X

        # Run PrivITP and account for actual realized cost
        t_processed, lam_tilde, is_correct = run_priv_itp(
            rewards_P, correct_P, beta, sigma, sigma_X, sigma_Z, L_trunc, n_cap, rng
        )

        eps_2_post_spent = compute_eps_2_post(
            lam=lam_tilde, R_max=R_max_P, Delta_r=Delta_r_P,
            beta=beta, sigma=sigma, t=t_processed, n_grid=n_grid,
        )
        spent = eps_2_post_spent + Delta_r_P / sigma_X

        ebudget -= spent
        if ebudget < 0:
            break

        eps_max_array.append(eps_max_P)
        stopping_time_array.append(t_processed)
        correct_array.append(is_correct)
        counter += 1

    eps_max_array = np.asarray(eps_max_array, dtype=np.float64)
    stopping_time_array = np.asarray(stopping_time_array, dtype=np.int32)
    correct_array = np.asarray(correct_array, dtype=np.int32)

    if len(eps_max_array) > 0:
        E_MAX = float(np.max(eps_max_array))
        T = ebudget_init / E_MAX
        avg_stopping_time = float(np.mean(stopping_time_array))
        accuracy = float(np.mean(correct_array))
    else:
        E_MAX = float("nan")
        T = float("nan")
        avg_stopping_time = float("nan")
        accuracy = float("nan")

    return {
        "counter": counter,
        "E_MAX": E_MAX,
        "T": T,
        "avg_stopping_time": avg_stopping_time,
        "accuracy": accuracy,
        "ebudget_left": ebudget,
    }


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    configure_matplotlib(grid=False)

    print(f"Loading {args.input_file} ...")
    all_data = load_scored_data(args.input_file)
    print(f"Loaded {len(all_data)} prompts.\n")

    n_sigmas = len(args.sigmas)
    n_seeds = len(args.seeds)

    counters_mat = np.zeros((n_sigmas, n_seeds))
    avg_st_mat = np.zeros((n_sigmas, n_seeds))
    avg_corr_mat = np.zeros((n_sigmas, n_seeds))
    T_mat = np.zeros((n_sigmas, n_seeds))

    for si, sigma in enumerate(args.sigmas):
        for ki, seed in enumerate(args.seeds):
            out = run_privacy_budget_loop(
                all_data,
                beta=args.beta,
                sigma=sigma,
                sigma_X=sigma / 2.0,
                sigma_Z=sigma / 2.0,
                L_trunc=args.truncation_L,
                n_cap=args.n_cap,
                ebudget_init=args.ebudget,
                seed=seed,
                n_grid=args.n_grid,
            )
            counters_mat[si, ki] = out["counter"]
            avg_st_mat[si, ki] = out["avg_stopping_time"]
            avg_corr_mat[si, ki] = out["accuracy"]
            T_mat[si, ki] = (
                int(np.floor(out["T"])) if not np.isnan(out["T"]) else np.nan
            )
        print(f"sigma={sigma} done.")

    counters_mean = np.nanmean(counters_mat, axis=1)
    counters_std = np.nanstd(counters_mat, axis=1)
    avg_st_mean = np.nanmean(avg_st_mat, axis=1)
    avg_st_std = np.nanstd(avg_st_mat, axis=1)
    avg_corr_mean = np.nanmean(avg_corr_mat, axis=1)
    avg_corr_std = np.nanstd(avg_corr_mat, axis=1)
    T_mean = np.nanmean(T_mat, axis=1)

    print("\n=== Aggregated (mean +/- std over seeds) ===")
    for si, sigma in enumerate(args.sigmas):
        print(
            f"sigma={sigma}: counter={counters_mean[si]:.1f}+/-{counters_std[si]:.1f} "
            f"avg_t={avg_st_mean[si]:.2f}+/-{avg_st_std[si]:.2f} "
            f"avg_correct={avg_corr_mean[si]:.3f}+/-{avg_corr_std[si]:.3f} "
            f"T_mean={T_mean[si]:.2f}"
        )

    # ----- Plot -----
    x = np.arange(n_sigmas)
    bar_width = 0.27

    fig, ax = plt.subplots(figsize=(10, 6))
    ax_right = ax.twinx()

    ax.bar(x - bar_width, counters_mean, bar_width,
           yerr=counters_std, capsize=5,
           color="#1f77b4", edgecolor="black",
           label="# Prompts processed",
           error_kw={"ecolor": "black", "elinewidth": 1.2})
    ax.bar(x, avg_st_mean, bar_width,
           yerr=avg_st_std, capsize=5,
           color="#2ca02c", edgecolor="black",
           label="Avg stopping time",
           error_kw={"ecolor": "black", "elinewidth": 1.2})
    ax_right.bar(x + bar_width, avg_corr_mean, bar_width,
                 yerr=avg_corr_std, capsize=5,
                 color="#ff7f0e", edgecolor="black",
                 label="Avg. accuracy",
                 error_kw={"ecolor": "black", "elinewidth": 1.2})

    # Basic-composition baseline as a short red dashed segment over each sigma's left bar
    basic_comp_handle = None
    for i, T_i in enumerate(T_mean):
        if not np.isnan(T_i):
            x_left = x[i] - bar_width - bar_width / 2
            x_right = x[i] - bar_width + bar_width / 2
            line = ax.hlines(T_i, x_left, x_right,
                             colors="red", linestyles="dashed", linewidth=2)
            if basic_comp_handle is None:
                basic_comp_handle = line

    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\sigma={s}$" for s in args.sigmas])
    ax.set_xlabel(r"Noise scale $\sigma$", fontsize=14)
    ax.set_ylabel("# prompts processed / avg stopping time", fontsize=14)
    ax_right.set_ylabel("avg accuracy", fontsize=14)
    ax_right.set_ylim(0, 1.0)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right
    if basic_comp_handle is not None:
        handles.append(basic_comp_handle)
        labels.append("basic composition")
    ax.legend(handles, labels, loc="best", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{args.output_basename}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    print(f"\nSaved: {args.output_basename}.png and {args.output_basename}.pdf")


if __name__ == "__main__":
    main()        "legend.framealpha": 0.92,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="PrivITP privacy-budget sweep over sigma."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Scored data file. Accepts either JSONL "
                             "(one record per line) or a JSON array.")
    parser.add_argument("--output_basename", type=str, default="private_itp_sigma_sweep",
                        help="Output filename without extension.")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Pessimism strength.")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[1.0, 5.0, 8.0, 10.0],
                        help="Total noise budgets to sweep over. Each is split equally "
                             "between Phase 1 (lambda) and Phase 2 (rewards).")
    parser.add_argument("--n_cap", type=int, default=16,
                        help="Maximum candidates considered per phase per prompt.")
    parser.add_argument("--ebudget", type=float, default=50.0,
                        help="Total privacy budget (cumulative epsilon).")
    parser.add_argument("--truncation_L", type=float, default=0.5,
                        help="Truncation constant L used in M = (R_max + sigma_Z*L - lambda)/beta.")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(10)),
                        help="Seeds to average over.")
    parser.add_argument("--n_grid", type=int, default=512,
                        help="Grid size for the numerical expectation in the accountant.")
    return parser.parse_args()


# ==========================================
# Algorithm primitives
# ==========================================
def compute_norm_constant(proxy_rewards, beta):
    N = len(proxy_rewards)
    sorted_rewards = np.sort(proxy_rewards)
    r_prev = -np.inf
    J = np.sum(sorted_rewards) / N
    Z = 1.0
    lam = (J - beta) / Z
    for i in range(N):
        lam = (J - beta) / Z
        r_curr = sorted_rewards[i]
        if (r_prev <= lam < r_curr) or (i == N - 1):
            return lam
        J -= r_curr / N
        Z -= 1.0 / N
        r_prev = r_curr
    return lam


def _log_expected_phi(arg_at_grid):
    """Stable log of the average of standard-normal CDFs over a grid of arguments."""
    log_phi = norm.logcdf(arg_at_grid)
    m = np.max(log_phi)
    return m + np.log(np.mean(np.exp(log_phi - m)))


def compute_eps_2_post(lam, R_max, Delta_r, beta, sigma, t, n_grid=512):
    """
    Per-prompt Phase-2 RDP accountant for PrivITP.

    Returns the log-ratio bound on epsilon_2 given the realized stopping time t,
    the noisy threshold lam, the prompt-specific R_max and Delta_r, and the
    Phase-2 noise scale sigma.
    """
    M = (R_max - lam) / beta
    u = (np.arange(n_grid) + 0.5) / n_grid
    bMu = beta * M * u

    arg_num_1 = (lam + bMu - R_max + Delta_r) / sigma
    arg_den_1 = (lam + bMu - R_max) / sigma
    log_ratio_1 = _log_expected_phi(arg_num_1) - _log_expected_phi(arg_den_1)

    arg_num_2 = (Delta_r - lam - bMu) / sigma
    arg_den_2 = (-lam - bMu) / sigma
    log_ratio_2 = _log_expected_phi(arg_num_2) - _log_expected_phi(arg_den_2)

    return (t - 1) * log_ratio_1 + log_ratio_2


def run_priv_itp(prompt_rewards, prompt_correct, beta, sigma,
                 sigma_X, sigma_Z, L_trunc, n_cap, rng):
    """Run a single sequential PrivITP pass on one prompt and return (t, lam_tilde, is_correct)."""
    R_max_full = float(np.max(prompt_rewards))

    # Phase 1: compute lambda on a clean subset, then perturb by N(0, sigma_X)
    idx_phase1 = rng.choice(len(prompt_rewards), size=n_cap, replace=False)
    r_phase1 = prompt_rewards[idx_phase1]
    lam = compute_norm_constant(r_phase1, beta)
    lam_tilde = lam + rng.normal(0.0, sigma_X)

    M = (R_max_full + sigma_Z * L_trunc - lam_tilde) / beta

    # Phase 2: noisy rejection sampling on a fresh subset
    idx_phase2 = rng.choice(len(prompt_rewards), size=n_cap, replace=False)
    r_phase2_clean = prompt_rewards[idx_phase2]
    r_phase2_noisy = r_phase2_clean + rng.normal(0.0, sigma, size=n_cap)
    correct_phase2 = prompt_correct[idx_phase2]

    t_processed = n_cap
    accepted_idx = n_cap - 1
    for i in range(n_cap):
        w_i = max((r_phase2_noisy[i] - lam_tilde) / beta, 0.0)
        p_i = min(w_i / M, 1.0) if M > 0 else 0.0
        if rng.random() < p_i:
            t_processed = i + 1
            accepted_idx = i
            break

    is_correct = int(correct_phase2[accepted_idx])
    return t_processed, lam_tilde, is_correct


# ==========================================
# Data loading
# ==========================================
def load_scored_data(path):
    """Load either a JSONL file or a JSON array of records."""
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


# ==========================================
# Main loop for one (sigma, seed)
# ==========================================
def run_privacy_budget_loop(all_data, beta, sigma, sigma_X, sigma_Z, L_trunc,
                            n_cap, ebudget_init, seed, n_grid):
    rng = np.random.default_rng(seed)

    # Filter prompts with enough candidates and stash (rewards, correct) pairs
    prompts_data = []
    for item in all_data:
        rewards, correct = [], []
        for r in item.get("responses", []):
            if r.get("proxy_reward") is not None:
                rewards.append(r["proxy_reward"])
                correct.append(r.get("is_correct", 0))
        if len(rewards) >= n_cap:
            prompts_data.append((np.asarray(rewards, dtype=np.float64),
                                 np.asarray(correct, dtype=np.int32)))

    # Global z-score normalization
    all_rewards_concat = np.concatenate([rw for rw, _ in prompts_data])
    global_mean = float(np.mean(all_rewards_concat))
    global_std = float(np.std(all_rewards_concat))
    if global_std == 0.0:
        global_std = 1.0
    prompts_data = [((rw - global_mean) / global_std, cc)
                    for rw, cc in prompts_data]

    order = rng.permutation(len(prompts_data))
    ebudget = ebudget_init
    counter = 0
    eps_max_array = []
    stopping_time_array = []
    correct_array = []

    for prompt_idx in order:
        rewards_P, correct_P = prompts_data[prompt_idx]
        R_max_P = float(np.max(rewards_P))
        Delta_r_P = R_max_P - float(np.min(rewards_P))

        # Worst-case accountant for this prompt (uses deterministic lambda)
        idx_emax = rng.choice(len(rewards_P), size=n_cap, replace=False)
        lam_det = compute_norm_constant(rewards_P[idx_emax], beta)
        eps_2_post_emax = compute_eps_2_post(
            lam=lam_det, R_max=R_max_P, Delta_r=Delta_r_P,
            beta=beta, sigma=sigma, t=n_cap, n_grid=n_grid,
        )
        eps_max_P = eps_2_post_emax + Delta_r_P / sigma_X

        # Run PrivITP and account for actual realized cost
        t_processed, lam_tilde, is_correct = run_priv_itp(
            rewards_P, correct_P, beta, sigma, sigma_X, sigma_Z, L_trunc, n_cap, rng
        )

        eps_2_post_spent = compute_eps_2_post(
            lam=lam_tilde, R_max=R_max_P, Delta_r=Delta_r_P,
            beta=beta, sigma=sigma, t=t_processed, n_grid=n_grid,
        )
        spent = eps_2_post_spent + Delta_r_P / sigma_X

        ebudget -= spent
        if ebudget < 0:
            break

        eps_max_array.append(eps_max_P)
        stopping_time_array.append(t_processed)
        correct_array.append(is_correct)
        counter += 1

    eps_max_array = np.asarray(eps_max_array, dtype=np.float64)
    stopping_time_array = np.asarray(stopping_time_array, dtype=np.int32)
    correct_array = np.asarray(correct_array, dtype=np.int32)

    if len(eps_max_array) > 0:
        E_MAX = float(np.max(eps_max_array))
        T = ebudget_init / E_MAX
        avg_stopping_time = float(np.mean(stopping_time_array))
        accuracy = float(np.mean(correct_array))
    else:
        E_MAX = float("nan")
        T = float("nan")
        avg_stopping_time = float("nan")
        accuracy = float("nan")

    return {
        "counter": counter,
        "E_MAX": E_MAX,
        "T": T,
        "avg_stopping_time": avg_stopping_time,
        "accuracy": accuracy,
        "ebudget_left": ebudget,
    }


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    configure_matplotlib()

    print(f"Loading {args.input_file} ...")
    all_data = load_scored_data(args.input_file)
    print(f"Loaded {len(all_data)} prompts.\n")

    n_sigmas = len(args.sigmas)
    n_seeds = len(args.seeds)

    counters_mat = np.zeros((n_sigmas, n_seeds))
    avg_st_mat = np.zeros((n_sigmas, n_seeds))
    avg_corr_mat = np.zeros((n_sigmas, n_seeds))
    T_mat = np.zeros((n_sigmas, n_seeds))

    for si, sigma in enumerate(args.sigmas):
        for ki, seed in enumerate(args.seeds):
            out = run_privacy_budget_loop(
                all_data,
                beta=args.beta,
                sigma=sigma,
                sigma_X=sigma / 2.0,
                sigma_Z=sigma / 2.0,
                L_trunc=args.truncation_L,
                n_cap=args.n_cap,
                ebudget_init=args.ebudget,
                seed=seed,
                n_grid=args.n_grid,
            )
            counters_mat[si, ki] = out["counter"]
            avg_st_mat[si, ki] = out["avg_stopping_time"]
            avg_corr_mat[si, ki] = out["accuracy"]
            T_mat[si, ki] = (
                int(np.floor(out["T"])) if not np.isnan(out["T"]) else np.nan
            )
        print(f"sigma={sigma} done.")

    counters_mean = np.nanmean(counters_mat, axis=1)
    counters_std = np.nanstd(counters_mat, axis=1)
    avg_st_mean = np.nanmean(avg_st_mat, axis=1)
    avg_st_std = np.nanstd(avg_st_mat, axis=1)
    avg_corr_mean = np.nanmean(avg_corr_mat, axis=1)
    avg_corr_std = np.nanstd(avg_corr_mat, axis=1)
    T_mean = np.nanmean(T_mat, axis=1)

    print("\n=== Aggregated (mean +/- std over seeds) ===")
    for si, sigma in enumerate(args.sigmas):
        print(
            f"sigma={sigma}: counter={counters_mean[si]:.1f}+/-{counters_std[si]:.1f} "
            f"avg_t={avg_st_mean[si]:.2f}+/-{avg_st_std[si]:.2f} "
            f"avg_correct={avg_corr_mean[si]:.3f}+/-{avg_corr_std[si]:.3f} "
            f"T_mean={T_mean[si]:.2f}"
        )

    # ----- Plot -----
    x = np.arange(n_sigmas)
    bar_width = 0.27

    fig, ax = plt.subplots(figsize=(10, 6))
    ax_right = ax.twinx()

    ax.bar(x - bar_width, counters_mean, bar_width,
           yerr=counters_std, capsize=5,
           color="#1f77b4", edgecolor="black",
           label="# Prompts processed",
           error_kw={"ecolor": "black", "elinewidth": 1.2})
    ax.bar(x, avg_st_mean, bar_width,
           yerr=avg_st_std, capsize=5,
           color="#2ca02c", edgecolor="black",
           label="Avg stopping time",
           error_kw={"ecolor": "black", "elinewidth": 1.2})
    ax_right.bar(x + bar_width, avg_corr_mean, bar_width,
                 yerr=avg_corr_std, capsize=5,
                 color="#ff7f0e", edgecolor="black",
                 label="Avg. accuracy",
                 error_kw={"ecolor": "black", "elinewidth": 1.2})

    # Basic-composition baseline as a short red dashed segment over each sigma's left bar
    basic_comp_handle = None
    for i, T_i in enumerate(T_mean):
        if not np.isnan(T_i):
            x_left = x[i] - bar_width - bar_width / 2
            x_right = x[i] - bar_width + bar_width / 2
            line = ax.hlines(T_i, x_left, x_right,
                             colors="red", linestyles="dashed", linewidth=2)
            if basic_comp_handle is None:
                basic_comp_handle = line

    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\sigma={s}$" for s in args.sigmas])
    ax.set_xlabel(r"Noise scale $\sigma$", fontsize=14)
    ax.set_ylabel("# prompts processed / avg stopping time", fontsize=14)
    ax_right.set_ylabel("avg accuracy", fontsize=14)
    ax_right.set_ylim(0, 1.0)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    handles = handles_left + handles_right
    labels = labels_left + labels_right
    if basic_comp_handle is not None:
        handles.append(basic_comp_handle)
        labels.append("basic composition")
    ax.legend(handles, labels, loc="best", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{args.output_basename}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    print(f"\nSaved: {args.output_basename}.png and {args.output_basename}.pdf")


if __name__ == "__main__":
    main()
