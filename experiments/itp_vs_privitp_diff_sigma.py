"""
ITP vs PrivITP with two-phase data splitting, swept over sigma.

For each prompt and each N, this script draws 2N candidates without replacement
and splits them into two disjoint halves of size N:
    phase1_idx -> used to compute the threshold lambda_hat
    phase2_idx -> used as the rejection-sampling pool

Standard ITP uses only phase 2 with a clean lambda computed on phase 2 (the
phase-1 pool is unused for ITP). PrivITP uses lambda computed on phase 1 with
N(0, sigma_X) noise added, plus N(0, sigma_Z) noise added to phase-2 rewards.
By default sigma_X = sigma_Z = sigma / 2 so the total noise budget across the
two phases equals sigma.

Outputs a single accuracy-lift plot comparing ITP against PrivITP at each
sigma in the sweep, plus a printed terminal table for each variant.
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
    parser = argparse.ArgumentParser(
        description="ITP vs PrivITP under two-phase data splitting, swept over sigma."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Scored JSONL (see top-level README for schema).")
    parser.add_argument("--output_basename", type=str, default="itp_vs_privitp_split",
                        help="Output filename without extension; PDF and PNG are written.")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Pessimism strength for ITP and PrivITP.")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.5, 1.5, 4.0],
                        help="Total noise budgets to sweep over. Each is split equally "
                             "between phase 1 (lambda) and phase 2 (rewards).")
    parser.add_argument("--n_max_exp", type=int, default=12,
                        help="N grid is [2^0, ..., 2^n_max_exp]. Each prompt needs at "
                             "least 2 * 2^n_max_exp candidates.")
    parser.add_argument("--m_replicates", type=int, default=50)
    parser.add_argument("--s_gauss", type=int, default=150,
                        help="Number of Gaussian noise samples for PrivITP.")
    parser.add_argument("--truncation_L", type=float, default=4.0,
                        help="Truncation constant L used in M_trunc = (R_max + sigma_Z * L - lambda)/beta.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rm_label", type=str, default="Reward Model")
    parser.add_argument("--dataset_label", type=str, default="Evaluation Set")
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
    """Non-private ITP on a single pool."""
    N = len(proxy_rewards)
    proxy_rewards = np.array(proxy_rewards)
    lam_hat = compute_norm_constant(proxy_rewards, beta)
    w = np.maximum((proxy_rewards - lam_hat) / beta, 0.0)
    M_trunc = max(np.max(w), 1e-8)
    p_accept = np.minimum(w / M_trunc, 1.0)

    probs = np.zeros(N)
    prob_no_one_accepted_yet = 1.0
    for i in range(N):
        probs[i] = prob_no_one_accepted_yet * p_accept[i]
        prob_no_one_accepted_yet *= (1.0 - p_accept[i])
    return probs, prob_no_one_accepted_yet


def get_privitp_two_phase(r_phase1, r_phase2, beta, sigma_X, sigma_Z,
                          L, num_samples):
    """
    Two-phase PrivITP with disjoint data splits.
        Phase 1: lambda_hat is computed on r_phase1 (clean), then perturbed
                 by N(0, sigma_X).
        Phase 2: r_phase2 rewards are perturbed by N(0, sigma_Z) for use in
                 rejection sampling. The truncation constant uses
                 R_max(r_phase2) + sigma_Z * L as the upper envelope.

    Acceptance probabilities are averaged over `num_samples` independent
    realizations before sequential rejection sampling.
    """
    n2 = len(r_phase2)
    r_phase2 = np.array(r_phase2)

    if sigma_X == 0 and sigma_Z == 0:
        return get_pessimism_probs(r_phase2, beta)

    lam_clean = compute_norm_constant(np.array(r_phase1), beta)
    g_lam = np.random.randn(num_samples) * sigma_X
    lam_noisy = lam_clean + g_lam

    R_max_phase2 = float(np.max(r_phase2))
    g_r = np.random.randn(n2, num_samples) * sigma_Z
    r_tilde = r_phase2[:, np.newaxis] + g_r

    p_acc_all = np.zeros((n2, num_samples))
    for s in range(num_samples):
        lam_s = lam_noisy[s]
        M_trunc = max((R_max_phase2 + sigma_Z * L - lam_s) / beta, 1e-8)
        w = np.maximum((r_tilde[:, s] - lam_s) / beta, 0.0)
        p_acc_all[:, s] = np.minimum(w / M_trunc, 1.0)

    p_accept_expected = np.mean(p_acc_all, axis=1)

    probs = np.zeros(n2)
    prob_no_one_accepted_yet = 1.0
    for i in range(n2):
        probs[i] = prob_no_one_accepted_yet * p_accept_expected[i]
        prob_no_one_accepted_yet *= (1.0 - p_accept_expected[i])
    return probs, prob_no_one_accepted_yet


def calc_metrics(probs, gold_rewards, proxy_rewards,
                 fallback_prob, base_gold, base_proxy):
    expected_true = np.sum(probs * np.array(gold_rewards)) + fallback_prob * base_gold
    expected_proxy = np.sum(probs * np.array(proxy_rewards)) + fallback_prob * base_proxy
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
    required_candidates = 2 * max(n_values)

    print(f"Loading scored dataset from {args.input_file}...")
    all_data = []
    with open(args.input_file, "r") as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"Total prompts: {len(all_data)}")
    print(f"Required candidates per prompt: {required_candidates} (2 * 2^{args.n_max_exp})")

    # ----- Global baselines (raw, no normalization) -----
    all_correct = []
    all_proxy = []
    for p in all_data:
        for r in p.get("responses", []):
            if r.get("proxy_reward") is not None:
                all_correct.append(r["is_correct"])
                all_proxy.append(r["proxy_reward"])

    base_gold = float(np.mean(all_correct))
    base_proxy = float(np.mean(all_proxy))
    print(f"Base accuracy: {base_gold * 100:.2f}%")
    print(f"Base mean proxy reward (raw): {base_proxy:.4f}")
    print(f"beta = {args.beta}, sigma sweep = {args.sigmas}\n")

    # ----- Result container -----
    results = {}
    for sigma in args.sigmas:
        results[("ITP", sigma)] = {n: {"true_r": [], "proxy_r": []} for n in n_values}
        results[("PrivITP", sigma)] = {n: {"true_r": [], "proxy_r": []} for n in n_values}

    # ----- Simulation -----
    print(f"Running ITP vs PrivITP across {len(args.sigmas)} sigma values "
          f"with {args.m_replicates} MC replicates each...")
    skipped = 0
    for item in tqdm(all_data, desc="Simulating prompts"):
        responses = [r for r in item.get("responses", [])
                     if r.get("proxy_reward") is not None]
        num_candidates = len(responses)
        if num_candidates < required_candidates:
            skipped += 1
            continue

        sc_arr = np.array([r["is_correct"] for r in responses])
        sr_arr = np.array([r["proxy_reward"] for r in responses])

        for n in n_values:
            tmp_itp_t = {s: [] for s in args.sigmas}
            tmp_itp_p = {s: [] for s in args.sigmas}
            tmp_privitp_t = {s: [] for s in args.sigmas}
            tmp_privitp_p = {s: [] for s in args.sigmas}

            for _ in range(args.m_replicates):
                idx_all = np.random.choice(num_candidates, 2 * n, replace=False)
                phase1_idx = idx_all[:n]
                phase2_idx = idx_all[n:]

                prox_p1 = sr_arr[phase1_idx]
                prox_p2 = sr_arr[phase2_idx]
                gold_p2 = sc_arr[phase2_idx]

                # ITP runs once per replicate (independent of sigma)
                pes_probs, pes_fb = get_pessimism_probs(prox_p2, args.beta)
                itp_t, itp_p = calc_metrics(pes_probs, gold_p2, prox_p2, pes_fb,
                                            base_gold, base_proxy)

                for sigma in args.sigmas:
                    tmp_itp_t[sigma].append(itp_t)
                    tmp_itp_p[sigma].append(itp_p)

                    pp_probs, pp_fb = get_privitp_two_phase(
                        r_phase1=prox_p1,
                        r_phase2=prox_p2,
                        beta=args.beta,
                        sigma_X=sigma / 2,
                        sigma_Z=sigma / 2,
                        L=args.truncation_L,
                        num_samples=args.s_gauss,
                    )
                    pp_t, pp_p = calc_metrics(pp_probs, gold_p2, prox_p2, pp_fb,
                                              base_gold, base_proxy)
                    tmp_privitp_t[sigma].append(pp_t)
                    tmp_privitp_p[sigma].append(pp_p)

            for sigma in args.sigmas:
                results[("ITP", sigma)][n]["true_r"].append(np.mean(tmp_itp_t[sigma]))
                results[("ITP", sigma)][n]["proxy_r"].append(np.mean(tmp_itp_p[sigma]))
                results[("PrivITP", sigma)][n]["true_r"].append(np.mean(tmp_privitp_t[sigma]))
                results[("PrivITP", sigma)][n]["proxy_r"].append(np.mean(tmp_privitp_p[sigma]))

    if skipped:
        print(f"\nSkipped {skipped} prompts with fewer than {required_candidates} candidates.")

    # ----- Aggregation -----
    def get_stats(algo, sigma, metric, is_lift=False):
        key = (algo, sigma)
        means, std_errs = [], []
        num_prompts = len(results[key][n_values[0]][metric])
        for n in n_values:
            vals = np.array(results[key][n][metric])
            if is_lift:
                means.append((np.mean(vals) - base_gold) * 100)
                std_errs.append((np.std(vals) / np.sqrt(num_prompts)) * 100)
            else:
                means.append(np.mean(vals))
                std_errs.append(np.std(vals) / np.sqrt(num_prompts))
        return np.array(means), np.array(std_errs)

    # ----- Plot -----
    print("\nGenerating plot and printing per-N statistics...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    sigma_colors = {sigma: palette[i % len(palette)] for i, sigma in enumerate(args.sigmas)}

    ax.axhline(0, color="grey", linestyle="-", linewidth=1.2, alpha=0.6,
               label=r"Base Policy ($\pi_{\mathrm{ref}}$)")

    # ITP baseline (same value across sigmas; use the first one)
    m_itp, se_itp = get_stats("ITP", args.sigmas[0], "true_r", is_lift=True)
    print(f"\n--- Baseline ITP (beta={args.beta}) ---")
    for i, n in enumerate(n_values):
        print(f"N={n:<4} | mean lift: {m_itp[i]:>6.2f}% | std err: {se_itp[i]:>5.2f}%")

    ax.plot(n_values, m_itp, color="black", linestyle="-", linewidth=2.8,
            label=fr"ITP ($\beta={args.beta}$)")
    ax.fill_between(n_values, m_itp - se_itp, m_itp + se_itp, color="black", alpha=0.12)

    # PrivITP variants
    for sigma in args.sigmas:
        m_priv, se_priv = get_stats("PrivITP", sigma, "true_r", is_lift=True)
        color = sigma_colors[sigma]
        print(f"\n--- PrivITP (sigma={sigma}, beta={args.beta}) ---")
        for i, n in enumerate(n_values):
            print(f"N={n:<4} | mean lift: {m_priv[i]:>6.2f}% | std err: {se_priv[i]:>5.2f}%")

        ax.plot(n_values, m_priv, color=color, linestyle="--", linewidth=2.8,
                label=fr"PrivITP ($\sigma={sigma}$)")
        ax.fill_between(n_values, m_priv - se_priv, m_priv + se_priv,
                        color=color, alpha=0.12)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values[::2])
    ax.set_xticklabels([fr"$2^{{{n}}}$" for n in range(0, args.n_max_exp + 1, 2)])
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"% Lift in Accuracy over $\pi_{\mathrm{ref}}$")
    ax.set_title("ITP vs PrivITP: Accuracy Lift", fontweight="bold")
    ax.legend(loc="best", ncol=2, handlelength=2.5, columnspacing=1.2)

    plt.tight_layout()
    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {args.output_basename}.pdf and {args.output_basename}.png")


if __name__ == "__main__":
    main()
