"""
Main comparison: BoN vs PrivBoN vs ITP vs PrivITP.

Evaluates four selection algorithms on a scored response dataset at the best
sigma and beta values found by the upstream hyperparameter sweeps:

    BoN      : Standard Best-of-N (argmax of proxy rewards).
    PrivBoN  : Gumbel-style softmax sampling with temperature sigma.
    ITP      : Pessimistic BoN with rejection sampling, parameter beta.
    PrivITP  : Two-phase private ITP. Phase 1 perturbs the threshold
               lambda_hat with N(0, sigma_phase); Phase 2 perturbs the
               rewards used in rejection sampling with an independent
               N(0, sigma_phase). Acceptance probabilities are averaged
               over Monte Carlo noise realizations.

Outputs both a two-panel figure (accuracy lift, expected proxy reward) and a
printed table of per-N statistics for each algorithm.
"""

import argparse
import json
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from plot_utils import configure_matplotlib

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Main comparison: BoN vs PrivBoN vs ITP vs PrivITP."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Scored JSONL (see top-level README for schema).")
    parser.add_argument("--output_basename", type=str, default="comparison_results",
                        help="Output filename without extension; PDF and PNG are written.")
    parser.add_argument("--sigma_gumbel", type=float, default=1.0,
                        help="Best sigma from the PrivBoN-Gumbel sweep "
                             "(in normalized reward space).")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Best beta from the ITP sweep "
                             "(in normalized reward space).")
    parser.add_argument("--sigma_phase", type=float, default=None,
                        help="Per-phase noise std for PrivITP. "
                             "Defaults to sigma_gumbel / 2 (so the two-phase noise budget "
                             "matches the single-phase Gumbel budget).")
    parser.add_argument("--n_max_exp", type=int, default=12,
                        help="N grid is [2^0, ..., 2^n_max_exp].")
    parser.add_argument("--m_replicates", type=int, default=50,
                        help="Monte Carlo replicates per (prompt, N).")
    parser.add_argument("--s_gauss", type=int, default=150,
                        help="Number of two-phase Gaussian noise samples for PrivITP.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rm_label", type=str, default="Reward Model")
    parser.add_argument("--dataset_label", type=str, default="Evaluation Set")
    return parser.parse_args()


# ==========================================
# Algorithm primitives
# ==========================================
def get_standard_bon_probs(proxy_rewards):
    probs = np.zeros(len(proxy_rewards))
    probs[np.argmax(proxy_rewards)] = 1.0
    return probs


def get_gumbel_probs(proxy_rewards, sigma):
    """Softmax over rewards with temperature sigma (the marginal of Gumbel-max)."""
    if sigma == 0:
        return get_standard_bon_probs(proxy_rewards)
    scaled = np.array(proxy_rewards) / sigma
    scaled -= np.max(scaled)
    exp_r = np.exp(scaled)
    return exp_r / np.sum(exp_r)


def compute_norm_constant(proxy_rewards, beta):
    """Solve (1/N) sum relu((r_i - lambda)/beta) = 1 for lambda."""
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
    """Non-private ITP: deterministic rejection sampling using lambda_hat."""
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


def get_private_pessimism_two_phase(proxy_rewards, beta, sigma_phase, num_samples):
    """
    Two-phase PrivITP.
        Phase 1: lambda_hat is computed on clean rewards, then perturbed by
                 N(0, sigma_phase) (a scalar per Monte Carlo sample).
        Phase 2: rewards used in rejection sampling are perturbed by an
                 independent N(0, sigma_phase) per element and per sample.

    Acceptance probabilities are averaged across `num_samples` realizations
    before the sequential rejection-sampling distribution is computed.
    """
    N = len(proxy_rewards)
    proxy_rewards = np.array(proxy_rewards)

    if sigma_phase == 0:
        return get_pessimism_probs(proxy_rewards, beta)

    # Phase 1: noisy lambda_hat
    lam_clean = compute_norm_constant(proxy_rewards, beta)
    g_lam = np.random.randn(num_samples) * sigma_phase
    lam_noisy = lam_clean + g_lam          # (num_samples,)

    # Phase 2: noisy rewards
    g_r = np.random.randn(N, num_samples) * sigma_phase
    noisy_r = proxy_rewards[:, np.newaxis] + g_r  # (N, num_samples)

    p_acc_all = np.zeros((N, num_samples))
    for s in range(num_samples):
        lam_s = lam_noisy[s]
        r_s = noisy_r[:, s]
        R_max = np.max(r_s)
        M_trunc = max((R_max - lam_s) / beta, 1e-8)
        w = np.maximum((r_s - lam_s) / beta, 0.0)
        p_acc_all[:, s] = np.minimum(w / M_trunc, 1.0)

    p_accept_expected = np.mean(p_acc_all, axis=1)

    probs = np.zeros(N)
    prob_no_one_accepted_yet = 1.0
    for i in range(N):
        probs[i] = prob_no_one_accepted_yet * p_accept_expected[i]
        prob_no_one_accepted_yet *= (1.0 - p_accept_expected[i])
    return probs, prob_no_one_accepted_yet


def calc_metrics(probs, gold_rewards, proxy_rewards_norm,
                 fallback_prob, base_gold, base_proxy_norm):
    expected_true = (np.sum(probs * np.array(gold_rewards))
                     + fallback_prob * base_gold)
    expected_proxy = (np.sum(probs * np.array(proxy_rewards_norm))
                      + fallback_prob * base_proxy_norm)
    return expected_true, expected_proxy


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    configure_matplotlib()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.sigma_phase is None:
        args.sigma_phase = args.sigma_gumbel / 2.0

    n_values = [2 ** i for i in range(args.n_max_exp + 1)]

    print(f"Loading scored dataset from {args.input_file}...")
    all_data = []
    with open(args.input_file, "r") as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"Total prompts: {len(all_data)}")

    # ----- Global reward normalization -----
    all_correct = []
    all_proxy_raw = []
    for p in all_data:
        for r in p.get("responses", []):
            if r.get("proxy_reward") is not None:
                all_correct.append(r["is_correct"])
                all_proxy_raw.append(r["proxy_reward"])

    all_proxy_raw = np.array(all_proxy_raw)
    GLOBAL_MEAN = float(np.mean(all_proxy_raw))
    GLOBAL_STD = float(np.std(all_proxy_raw))

    base_gold = float(np.mean(all_correct))
    base_proxy_norm = 0.0
    print(f"Base policy accuracy: {base_gold * 100:.2f}%\n")

    # ----- Simulation -----
    results = {
        "BoN":     {n: {"true_r": [], "proxy_r": []} for n in n_values},
        "PrivBoN": {n: {"true_r": [], "proxy_r": []} for n in n_values},
        "ITP":     {n: {"true_r": [], "proxy_r": []} for n in n_values},
        "PrivITP": {n: {"true_r": [], "proxy_r": []} for n in n_values},
    }

    for item in tqdm(all_data, desc="Simulating prompts"):
        responses = [r for r in item.get("responses", [])
                     if r.get("proxy_reward") is not None]
        num_candidates = len(responses)
        if num_candidates < max(n_values):
            continue

        sc_arr = np.array([r["is_correct"] for r in responses])
        sr_arr = (np.array([r["proxy_reward"] for r in responses])
                  - GLOBAL_MEAN) / GLOBAL_STD

        for n in n_values:
            tmp = {algo: {"t": [], "p": []} for algo in results}

            for _ in range(args.m_replicates):
                idx = np.random.choice(num_candidates, n, replace=False)
                prox_r = sr_arr[idx]
                gold_r = sc_arr[idx]

                # BoN
                p_bon = get_standard_bon_probs(prox_r)
                t, p = calc_metrics(p_bon, gold_r, prox_r, 0.0,
                                    base_gold, base_proxy_norm)
                tmp["BoN"]["t"].append(t); tmp["BoN"]["p"].append(p)

                # PrivBoN
                p_priv = get_gumbel_probs(prox_r, args.sigma_gumbel)
                t, p = calc_metrics(p_priv, gold_r, prox_r, 0.0,
                                    base_gold, base_proxy_norm)
                tmp["PrivBoN"]["t"].append(t); tmp["PrivBoN"]["p"].append(p)

                # ITP
                p_itp, fb_itp = get_pessimism_probs(prox_r, args.beta)
                t, p = calc_metrics(p_itp, gold_r, prox_r, fb_itp,
                                    base_gold, base_proxy_norm)
                tmp["ITP"]["t"].append(t); tmp["ITP"]["p"].append(p)

                # PrivITP
                p_priv_itp, fb_pp = get_private_pessimism_two_phase(
                    prox_r, args.beta, args.sigma_phase, num_samples=args.s_gauss
                )
                t, p = calc_metrics(p_priv_itp, gold_r, prox_r, fb_pp,
                                    base_gold, base_proxy_norm)
                tmp["PrivITP"]["t"].append(t); tmp["PrivITP"]["p"].append(p)

            for algo in results:
                results[algo][n]["true_r"].append(np.mean(tmp[algo]["t"]))
                results[algo][n]["proxy_r"].append(np.mean(tmp[algo]["p"]))

    # ----- Aggregation -----
    def get_stats(algo, metric, is_lift=False):
        means, std_errs = [], []
        num_prompts = len(results[algo][n_values[0]][metric])
        for n in n_values:
            prompt_means = np.array(results[algo][n][metric])
            if is_lift:
                means.append((np.mean(prompt_means) - base_gold) * 100)
                std_errs.append((np.std(prompt_means) / np.sqrt(num_prompts)) * 100)
            else:
                means.append(np.mean(prompt_means))
                std_errs.append(np.std(prompt_means) / np.sqrt(num_prompts))
        return np.array(means), np.array(std_errs)

    # ----- Plot configs -----
    plot_configs = [
        ("BoN",     "BoN",
         "black",   "X", "--", 2.5),
        ("PrivBoN", fr"PrivBoN ($\sigma={args.sigma_gumbel}$)",
         "#2ca02c", "s", "-",  2.0),
        ("ITP",     fr"ITP ($\beta={args.beta}$)",
         "#d62728", "^", "-",  2.0),
        ("PrivITP", fr"PrivITP ($\sigma_\phi={args.sigma_phase},\ \beta={args.beta}$)",
         "#9467bd", "D", "-.", 2.5),
    ]

    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Panel 0: accuracy lift -----
    ax = axes[0]
    ax.axhline(0, color="grey", linestyle="-", linewidth=1.5, alpha=0.5,
               label=r"Base Policy ($\pi_{\mathrm{ref}}$)")

    print("\n" + "=" * 55)
    print("ACCURACY LIFT RESULTS (mean +/- std error, %)")
    print("=" * 55)

    for algo, label, color, marker, ls, lw in plot_configs:
        m, se = get_stats(algo, "true_r", is_lift=True)
        print(f"\n--- {algo} ---")
        for i, n in enumerate(n_values):
            print(f"N={n:<5} | mean lift: {m[i]:>6.2f}% | std err: {se[i]:>5.2f}%")
        ax.plot(n_values, m, color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=6, label=label)
        ax.fill_between(n_values, m - se, m + se, color=color, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values[::2])
    ax.set_xticklabels([fr"$2^{{{n}}}$" for n in range(0, args.n_max_exp + 1, 2)])
    ax.set_xlabel("$N$", fontsize=12)
    ax.set_ylabel(r"% Lift in Accuracy over $\pi_{\mathrm{ref}}$", fontsize=12)
    ax.set_title("Accuracy Lift", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best")

    # ----- Panel 1: estimated reward -----
    ax = axes[1]
    ax.axhline(base_proxy_norm, color="grey", linestyle=":", linewidth=1.5, alpha=0.5,
               label=r"$\pi_{\mathrm{ref}}$ mean")

    print("\n" + "=" * 55)
    print("EXPECTED PROXY REWARD RESULTS (mean +/- std error)")
    print("=" * 55)

    for algo, label, color, marker, ls, lw in plot_configs:
        m, se = get_stats(algo, "proxy_r", is_lift=False)
        print(f"\n--- {algo} ---")
        for i, n in enumerate(n_values):
            print(f"N={n:<5} | mean proxy: {m[i]:>7.4f} | std err: {se[i]:>6.4f}")
        ax.plot(n_values, m, color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=6, label=label)
        ax.fill_between(n_values, m - se, m + se, color=color, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values[::2])
    ax.set_xticklabels([fr"$2^{{{n}}}$" for n in range(0, args.n_max_exp + 1, 2)])
    ax.set_xlabel("$N$", fontsize=12)
    ax.set_ylabel("Expected Proxy Reward", fontsize=12)
    ax.set_title("Estimated Reward", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {args.output_basename}.pdf and {args.output_basename}.png")


if __name__ == "__main__":
    main()
