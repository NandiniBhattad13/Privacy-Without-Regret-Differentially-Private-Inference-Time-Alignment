"""
Pessimistic Best-of-N: hyperparameter sweep over beta.

Loads a scored JSONL (output of the dataset-generation stage) and, for each
prompt, simulates Standard BoN and Pessimistic BoN at several beta values
across an exponentially-spaced grid of N. Plots two panels:
    (left)  accuracy lift over the base policy
    (right) expected proxy reward (globally z-score normalized)
"""

import argparse
import json
import random
import warnings


from plot_utils import configure_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")




def parse_args():
    parser = argparse.ArgumentParser(description="Pessimistic BoN beta sweep.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Scored JSONL produced by the dataset-generation stage.")
    parser.add_argument("--output_basename", type=str, default="beta_sweep_results",
                        help="Output filename without extension; PDF and PNG are written.")
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                        help="Beta values to sweep over (in normalized reward space).")
    parser.add_argument("--n_max_exp", type=int, default=13,
                        help="N grid is [2^0, ..., 2^n_max_exp].")
    parser.add_argument("--m_replicates", type=int, default=50,
                        help="Monte Carlo replicates per (prompt, N).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rm_label", type=str, default="Reward Model",
                        help="Label for the reward model used (only affects plot title).")
    parser.add_argument("--dataset_label", type=str, default="Evaluation Set",
                        help="Label for the dataset used (only affects plot title).")
    return parser.parse_args()


# ==========================================
# Probability and metric functions
# ==========================================
def get_standard_bon_probs(proxy_rewards):
    probs = np.zeros(len(proxy_rewards))
    probs[np.argmax(proxy_rewards)] = 1.0
    return probs


def compute_norm_constant(proxy_rewards, beta):
    N = len(proxy_rewards)
    sorted_rewards = np.sort(proxy_rewards)
    r_prev = -np.inf
    J = np.sum(sorted_rewards) / N
    Z = 1.0
    lam = None
    for i in range(N):
        lam = (J - beta) / Z
        r_curr = sorted_rewards[i]
        if (r_prev <= lam < r_curr) or (i == N - 1):
            return lam
        J -= r_curr / N
        Z -= 1.0 / N
        r_prev = r_curr
    return lam


def get_pessimism_probs(proxy_rewards, beta):
    N = len(proxy_rewards)
    lam_hat = compute_norm_constant(proxy_rewards, beta)
    w = np.maximum((np.array(proxy_rewards) - lam_hat) / beta, 0.0)
    M_trunc = max(np.max(w), 1e-8)
    p_accept = np.minimum(w / M_trunc, 1.0)

    probs = np.zeros(N)
    prob_no_one_accepted_yet = 1.0
    for i in range(N):
        probs[i] = prob_no_one_accepted_yet * p_accept[i]
        prob_no_one_accepted_yet *= (1.0 - p_accept[i])
    return probs, prob_no_one_accepted_yet


def calc_metrics(probs, gold_rewards, proxy_rewards_norm,
                 fallback_prob, base_gold, base_proxy_norm):
    expected_true = np.sum(probs * np.array(gold_rewards)) + (fallback_prob * base_gold)
    expected_proxy = np.sum(probs * np.array(proxy_rewards_norm)) + (fallback_prob * base_proxy_norm)
    return expected_true, expected_proxy


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    configure_matplotlib()

    random.seed(args.seed)
    np.random.seed(args.seed)

    n_values = [2 ** i for i in range(args.n_max_exp + 1)]

    print(f"Loading scored dataset from {args.input_file}...")
    all_data = []
    with open(args.input_file, "r") as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(all_data)} prompts.")

    # ----- Global reward normalization -----
    all_correct = []
    all_proxy_raw = []
    for p in all_data:
        for r in p.get("responses", []):
            if r.get("proxy_reward") is not None:
                all_correct.append(r["is_correct"])
                all_proxy_raw.append(r["proxy_reward"])

    all_proxy_raw = np.array(all_proxy_raw)
    GLOBAL_MEAN = np.mean(all_proxy_raw)
    GLOBAL_STD = np.std(all_proxy_raw)

    print("\n=== Reward Normalization Statistics ===")
    print(f"Raw mean: {GLOBAL_MEAN:.6f}, std: {GLOBAL_STD:.6f}")
    print(f"Range: [{np.min(all_proxy_raw):.4f}, {np.max(all_proxy_raw):.4f}]")
    print("=========================================\n")

    base_gold = np.mean(all_correct)
    base_proxy_norm = 0.0
    print(f"Base policy accuracy: {base_gold * 100:.2f}%\n")

    # ----- Simulation -----
    results = {
        "bon": {n: {"true_r": [], "proxy_r": []} for n in n_values},
        "pessimism": {b: {n: {"true_r": [], "proxy_r": []} for n in n_values}
                      for b in args.betas},
    }

    print("Running pessimistic BoN sweep over beta...")
    for item in tqdm(all_data, desc="Simulating prompts"):
        responses = [r for r in item.get("responses", [])
                     if r.get("proxy_reward") is not None]
        num_candidates = len(responses)
        if num_candidates < max(n_values):
            continue

        sc_arr = np.array([r["is_correct"] for r in responses])
        sr_arr = (np.array([r["proxy_reward"] for r in responses]) - GLOBAL_MEAN) / GLOBAL_STD

        for n in n_values:
            tmp_bon_t, tmp_bon_p = [], []
            tmp_pes_t = {b: [] for b in args.betas}
            tmp_pes_p = {b: [] for b in args.betas}

            for _ in range(args.m_replicates):
                idx = np.random.choice(num_candidates, n, replace=False)
                prox_r = sr_arr[idx]
                gold_r = sc_arr[idx]

                std_bon_probs = get_standard_bon_probs(prox_r)
                bt, bp = calc_metrics(std_bon_probs, gold_r, prox_r, 0.0,
                                      base_gold, base_proxy_norm)
                tmp_bon_t.append(bt); tmp_bon_p.append(bp)

                for b in args.betas:
                    pes_probs, fb = get_pessimism_probs(prox_r, b)
                    pt, pp = calc_metrics(pes_probs, gold_r, prox_r, fb,
                                          base_gold, base_proxy_norm)
                    tmp_pes_t[b].append(pt); tmp_pes_p[b].append(pp)

            results["bon"][n]["true_r"].append(np.mean(tmp_bon_t))
            results["bon"][n]["proxy_r"].append(np.mean(tmp_bon_p))
            for b in args.betas:
                results["pessimism"][b][n]["true_r"].append(np.mean(tmp_pes_t[b]))
                results["pessimism"][b][n]["proxy_r"].append(np.mean(tmp_pes_p[b]))

    # ----- Aggregation -----
    def get_stats(data_subset, is_lift=False):
        means, std_errs = [], []
        num_prompts = len(data_subset[n_values[0]])
        for n in n_values:
            prompt_means = np.array(data_subset[n])
            if is_lift:
                means.append((np.mean(prompt_means) - base_gold) * 100)
                std_errs.append((np.std(prompt_means) / np.sqrt(num_prompts)) * 100)
            else:
                means.append(np.mean(prompt_means))
                std_errs.append(np.std(prompt_means) / np.sqrt(num_prompts))
        return np.array(means), np.array(std_errs)

    # ----- Plot -----
    print("\nGenerating beta-sweep dashboard...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(args.betas)))

    # Panel 0: accuracy lift
    ax = axes[0]
    ax.axhline(0, color="grey", linestyle="-", linewidth=1.2, alpha=0.6,
               label=r"Base Policy ($\pi_{\mathrm{ref}}$)")

    bon_m, bon_se = get_stats({n: results["bon"][n]["true_r"] for n in n_values}, is_lift=True)
    ax.plot(n_values, bon_m, color="black", linestyle="--", linewidth=3.0,
            label=r"BoN ($\beta=0$)")
    ax.fill_between(n_values, bon_m - bon_se, bon_m + bon_se, color="black", alpha=0.12)

    for idx, b in enumerate(args.betas):
        m, se = get_stats({n: results["pessimism"][b][n]["true_r"] for n in n_values},
                          is_lift=True)
        ax.plot(n_values, m, color=colors[idx], linestyle="-", linewidth=2.3,
                label=fr"$\beta={b}$")
        ax.fill_between(n_values, m - se, m + se, color=colors[idx], alpha=0.10)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values[::2])
    ax.set_xticklabels([fr"$2^{{{n}}}$" for n in range(0, args.n_max_exp + 1, 2)])
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"% Lift in Accuracy over $\pi_{\mathrm{ref}}$")
    ax.set_title("Accuracy Lift", fontweight="bold")
    ax.legend(loc="best", ncol=2, handlelength=2.5, columnspacing=1.2)

    # Panel 1: estimated reward
    ax = axes[1]
    ax.axhline(base_proxy_norm, color="grey", linestyle="-", linewidth=1.2, alpha=0.6,
               label=r"Base Policy Mean")

    bon_p_m, bon_p_se = get_stats({n: results["bon"][n]["proxy_r"] for n in n_values}, is_lift=False)
    ax.plot(n_values, bon_p_m, color="black", linestyle="--", linewidth=3.0,
            label=r"BoN ($\beta=0$)")
    ax.fill_between(n_values, bon_p_m - bon_p_se, bon_p_m + bon_p_se, color="black", alpha=0.12)

    for idx, b in enumerate(args.betas):
        m, se = get_stats({n: results["pessimism"][b][n]["proxy_r"] for n in n_values},
                          is_lift=False)
        ax.plot(n_values, m, color=colors[idx], linestyle="-", linewidth=2.3,
                label=fr"$\beta={b}$")
        ax.fill_between(n_values, m - se, m + se, color=colors[idx], alpha=0.10)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values[::2])
    ax.set_xticklabels([fr"$2^{{{n}}}$" for n in range(0, args.n_max_exp + 1, 2)])
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"Expected Proxy Reward $\mathbb{E}[\hat{r}]$ (Normalized)")
    ax.set_title("Estimated Reward", fontweight="bold")

    plt.suptitle(
        f"Pessimistic BoN Evaluation ({args.dataset_label} | {args.rm_label} | Normalized)",
        fontsize=18, fontweight="bold", y=1.05,
    )
    plt.tight_layout()

    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output_basename}.pdf and {args.output_basename}.png")


if __name__ == "__main__":
    main()
