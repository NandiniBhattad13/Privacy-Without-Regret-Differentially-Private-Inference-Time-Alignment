"""
Private Best-of-N: hyperparameter sweep over noise level sigma.

For each prompt in the scored dataset, this script simulates Standard BoN and
two Private BoN variants (Gumbel softmax sampling and Gaussian-noise argmax)
across a grid of (N, sigma). It then plots a 2x3 panel grid (one panel per
sigma). Two metrics are supported:

    --metric accuracy_lift     (% lift in MMLU correctness over base policy)
    --metric estimated_reward  (expected proxy reward, z-score normalized)
"""

import argparse
import json
import warnings


from plot_utils import configure_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")




def parse_args():
    parser = argparse.ArgumentParser(description="Private BoN sigma sweep.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Scored JSONL (see top-level README for schema).")
    parser.add_argument("--output_basename", type=str, default="sigma_sweep_results",
                        help="Output filename without extension; PDF and PNG are written.")
    parser.add_argument("--metric", type=str, required=True,
                        choices=["accuracy_lift", "estimated_reward"],
                        help="Which metric to plot.")
    parser.add_argument("--sigmas", type=float, nargs="+", default=None,
                        help="Sigma values. Defaults: [0.5,0.75,1,1.25,1.5,2] for accuracy_lift, "
                             "[0.1,0.2,0.5,1,1.5,2] for estimated_reward.")
    parser.add_argument("--n_max_exp", type=int, default=12,
                        help="N grid is [2^0, ..., 2^n_max_exp].")
    parser.add_argument("--m_replicates", type=int, default=50,
                        help="Monte Carlo replicates per (prompt, N).")
    parser.add_argument("--s_gauss", type=int, default=150,
                        help="Number of Gaussian-noise samples per (prompt, N) replicate.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--y_min", type=float, default=None,
                        help="Override y-axis lower bound on plots.")
    parser.add_argument("--y_max", type=float, default=None,
                        help="Override y-axis upper bound on plots.")
    parser.add_argument("--rm_label", type=str, default="Reward Model")
    parser.add_argument("--dataset_label", type=str, default="Evaluation Set")
    return parser.parse_args()


def default_sigmas(metric):
    if metric == "accuracy_lift":
        return [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    return [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]


# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    configure_matplotlib()

    if args.seed is not None:
        np.random.seed(args.seed)

    sigmas = args.sigmas if args.sigmas is not None else default_sigmas(args.metric)
    n_values = [2 ** i for i in range(args.n_max_exp + 1)]

    print(f"Loading scored dataset from {args.input_file}...")
    data = []
    with open(args.input_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} prompts.")

    # Base accuracy 
    all_correct = []
    for p in data:
        for r in p.get("responses", []):
            if r.get("proxy_reward") is not None:
                all_correct.append(r["is_correct"])
    baseline_acc = float(np.mean(all_correct))
    if args.metric == "accuracy_lift":
        print(f"Base policy accuracy: {baseline_acc * 100:.2f}%")

    # ----- Result containers -----
    bon_results = {n: [] for n in n_values}
    gumbel_results = {s: {n: [] for n in n_values} for s in sigmas}
    gauss_results = {s: {n: [] for n in n_values} for s in sigmas}

    # ----- Simulation -----
    desc = f"Simulating Private BoN ({args.metric})"
    for prompt_data in tqdm(data, desc=desc):
        responses = [r for r in prompt_data.get("responses", [])
                     if r.get("proxy_reward") is not None]
        num_candidates = len(responses)
        if num_candidates < 2:
            continue

        sc_arr = np.array([r["is_correct"] for r in responses])
        sr_arr = np.array([r["proxy_reward"] for r in responses])

        # Per-prompt z-score normalization 
        std_dev = np.std(sr_arr)
        if std_dev > 0:
            sr_arr = (sr_arr - np.mean(sr_arr)) / std_dev
        else:
            sr_arr = sr_arr - np.mean(sr_arr)

        for n in n_values:
            if num_candidates < n:
                continue

            b_vals = []
            g_vals = {s: [] for s in sigmas}
            ga_vals = {s: [] for s in sigmas}

            base_noise = np.random.normal(0.0, 1.0, size=(args.s_gauss, n))

            for _ in range(args.m_replicates):
                idx = np.random.choice(num_candidates, n, replace=False)
                sr = sr_arr[idx]
                sc = sc_arr[idx]

                # Standard BoN
                if args.metric == "accuracy_lift":
                    b_vals.append(sc[np.argmax(sr)])
                else:
                    b_vals.append(np.max(sr))

                for s in sigmas:
                    # Gumbel-style softmax sampling distribution
                    r_scaled = sr / s
                    r_scaled -= np.max(r_scaled)
                    pi_g = np.exp(r_scaled)
                    pi_g /= np.sum(pi_g)

                    # Gaussian-noise argmax distribution (Monte Carlo)
                    r_noisy = sr + (base_noise * s)
                    winners_ga = np.argmax(r_noisy, axis=1)
                    pi_ga = np.bincount(winners_ga, minlength=n) / args.s_gauss

                    if args.metric == "accuracy_lift":
                        g_vals[s].append(np.sum(pi_g * sc))
                        ga_vals[s].append(np.sum(pi_ga * sc))
                    else:
                        g_vals[s].append(np.sum(pi_g * sr))
                        ga_vals[s].append(np.sum(pi_ga * sr))

            bon_results[n].append(np.mean(b_vals))
            for s in sigmas:
                gumbel_results[s][n].append(np.mean(g_vals[s]))
                gauss_results[s][n].append(np.mean(ga_vals[s]))

    # ----- Aggregation -----
    valid_n = [n for n in n_values if bon_results[n]]

    if args.metric == "accuracy_lift":
        def get_stats(acc_dict):
            means, ses = [], []
            num_prompts = len(acc_dict[valid_n[0]])
            for n in valid_n:
                vals = np.array(acc_dict[n])
                means.append((np.mean(vals) - baseline_acc) * 100)
                ses.append((np.std(vals) / np.sqrt(num_prompts)) * 100)
            return np.array(means), np.array(ses)

        bon_m, bon_se = get_stats(bon_results)
    else:
        def get_means(d):
            return np.array([np.mean(d[n]) for n in valid_n])

        bon_m = get_means(bon_results)
        bon_se = None  # original estimated-reward script didn't compute SE bands

    # ----- Plot -----
    print("\nGenerating 2x3 grid plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    # Determine y-axis bounds
    if args.y_min is not None and args.y_max is not None:
        y_min, y_max = args.y_min, args.y_max
    elif args.metric == "accuracy_lift":
        y_min, y_max = -12, 32
    else:
        all_vals = list(bon_m)
        for s in sigmas:
            all_vals += list(get_means(gumbel_results[s]))
            all_vals += list(get_means(gauss_results[s]))
        y_min = min(all_vals) - 0.2
        y_max = max(all_vals) + 0.2

    for i, s in enumerate(sigmas):
        ax = axes[i]

        if args.metric == "accuracy_lift":
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
            gum_m, gum_se = get_stats(gumbel_results[s])
            gau_m, gau_se = get_stats(gauss_results[s])

            ax.plot(valid_n, bon_m, "k--", linewidth=2, alpha=0.6, label="BoN")
            ax.fill_between(valid_n, bon_m - bon_se, bon_m + bon_se,
                            color="black", alpha=0.1)

            ax.plot(valid_n, gau_m, marker="s", color="#2f855a",
                    linewidth=2, markersize=5, label="PrivBoN-Gaussian")
            ax.fill_between(valid_n, gau_m - gau_se, gau_m + gau_se,
                            color="#2f855a", alpha=0.2)

            ax.plot(valid_n, gum_m, marker="^", color="#dd6b20",
                    linewidth=2, markersize=5, label="PrivBoN-Gumbel")
            ax.fill_between(valid_n, gum_m - gum_se, gum_m + gum_se,
                            color="#dd6b20", alpha=0.2)

            ylabel = r"% Lift over $\pi_{\mathrm{ref}}$"
        else:
            gum_m = get_means(gumbel_results[s])
            gau_m = get_means(gauss_results[s])

            ax.plot(valid_n, bon_m, "k--", linewidth=2.5,
                    label=r"Standard BoN ($\sigma=0$)")
            ax.plot(valid_n, gum_m, marker="o", markersize=6, color="teal",
                    linewidth=2, label="PrivBoN-Gumbel")
            ax.plot(valid_n, gau_m, marker="s", markersize=6, color="salmon",
                    linewidth=2, label="PrivBoN-Gaussian")

            ylabel = r"Expected Proxy Reward $\mathbb{E}[\hat{r}]$"

        ax.set_xscale("log", base=2)
        ax.set_xticks(valid_n[::2])
        ax.set_xticklabels([fr"$2^{{{int(np.log2(n))}}}$" for n in valid_n[::2]])
        ax.set_ylim(y_min, y_max)
        ax.set_title(fr"Noise Level: $\sigma={s}$", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

        if i >= 3:
            ax.set_xlabel("Number of Candidates ($N$)", fontsize=13)
        if i % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        if i == 0:
            ax.legend(fontsize=10, loc="upper left")

    title_metric = "Accuracy Lift" if args.metric == "accuracy_lift" else "Estimated Reward"
    plt.suptitle(
        f"BoN vs Private BoN: {title_metric}\n({args.dataset_label} | {args.rm_label} | Normalized)",
        fontsize=18, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output_basename}.pdf and {args.output_basename}.png")


if __name__ == "__main__":
    main()def main():
    args = parse_args()
    configure_matplotlib()

    if args.seed is not None:
        np.random.seed(args.seed)

    sigmas = args.sigmas if args.sigmas is not None else default_sigmas(args.metric)
    n_values = [2 ** i for i in range(args.n_max_exp + 1)]

    print(f"Loading scored dataset from {args.input_file}...")
    data = []
    with open(args.input_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} prompts.")

    # Base accuracy 
    all_correct = []
    for p in data:
        for r in p.get("responses", []):
            if r.get("proxy_reward") is not None:
                all_correct.append(r["is_correct"])
    baseline_acc = float(np.mean(all_correct))
    if args.metric == "accuracy_lift":
        print(f"Base policy accuracy: {baseline_acc * 100:.2f}%")

    # ----- Result containers -----
    bon_results = {n: [] for n in n_values}
    gumbel_results = {s: {n: [] for n in n_values} for s in sigmas}
    gauss_results = {s: {n: [] for n in n_values} for s in sigmas}

    # ----- Simulation -----
    desc = f"Simulating Private BoN ({args.metric})"
    for prompt_data in tqdm(data, desc=desc):
        responses = [r for r in prompt_data.get("responses", [])
                     if r.get("proxy_reward") is not None]
        num_candidates = len(responses)
        if num_candidates < 2:
            continue

        sc_arr = np.array([r["is_correct"] for r in responses])
        sr_arr = np.array([r["proxy_reward"] for r in responses])

        # Per-prompt z-score normalization 
        std_dev = np.std(sr_arr)
        if std_dev > 0:
            sr_arr = (sr_arr - np.mean(sr_arr)) / std_dev
        else:
            sr_arr = sr_arr - np.mean(sr_arr)

        for n in n_values:
            if num_candidates < n:
                continue

            b_vals = []
            g_vals = {s: [] for s in sigmas}
            ga_vals = {s: [] for s in sigmas}

            base_noise = np.random.normal(0.0, 1.0, size=(args.s_gauss, n))

            for _ in range(args.m_replicates):
                idx = np.random.choice(num_candidates, n, replace=False)
                sr = sr_arr[idx]
                sc = sc_arr[idx]

                # Standard BoN
                if args.metric == "accuracy_lift":
                    b_vals.append(sc[np.argmax(sr)])
                else:
                    b_vals.append(np.max(sr))

                for s in sigmas:
                    # Gumbel-style softmax sampling distribution
                    r_scaled = sr / s
                    r_scaled -= np.max(r_scaled)
                    pi_g = np.exp(r_scaled)
                    pi_g /= np.sum(pi_g)

                    # Gaussian-noise argmax distribution (Monte Carlo)
                    r_noisy = sr + (base_noise * s)
                    winners_ga = np.argmax(r_noisy, axis=1)
                    pi_ga = np.bincount(winners_ga, minlength=n) / args.s_gauss

                    if args.metric == "accuracy_lift":
                        g_vals[s].append(np.sum(pi_g * sc))
                        ga_vals[s].append(np.sum(pi_ga * sc))
                    else:
                        g_vals[s].append(np.sum(pi_g * sr))
                        ga_vals[s].append(np.sum(pi_ga * sr))

            bon_results[n].append(np.mean(b_vals))
            for s in sigmas:
                gumbel_results[s][n].append(np.mean(g_vals[s]))
                gauss_results[s][n].append(np.mean(ga_vals[s]))

    # ----- Aggregation -----
    valid_n = [n for n in n_values if bon_results[n]]

    if args.metric == "accuracy_lift":
        def get_stats(acc_dict):
            means, ses = [], []
            num_prompts = len(acc_dict[valid_n[0]])
            for n in valid_n:
                vals = np.array(acc_dict[n])
                means.append((np.mean(vals) - baseline_acc) * 100)
                ses.append((np.std(vals) / np.sqrt(num_prompts)) * 100)
            return np.array(means), np.array(ses)

        bon_m, bon_se = get_stats(bon_results)
    else:
        def get_means(d):
            return np.array([np.mean(d[n]) for n in valid_n])

        bon_m = get_means(bon_results)
        bon_se = None  # original estimated-reward script didn't compute SE bands

    # ----- Plot -----
    print("\nGenerating 2x3 grid plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()

    # Determine y-axis bounds
    if args.y_min is not None and args.y_max is not None:
        y_min, y_max = args.y_min, args.y_max
    elif args.metric == "accuracy_lift":
        y_min, y_max = -12, 32
    else:
        all_vals = list(bon_m)
        for s in sigmas:
            all_vals += list(get_means(gumbel_results[s]))
            all_vals += list(get_means(gauss_results[s]))
        y_min = min(all_vals) - 0.2
        y_max = max(all_vals) + 0.2

    for i, s in enumerate(sigmas):
        ax = axes[i]

        if args.metric == "accuracy_lift":
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
            gum_m, gum_se = get_stats(gumbel_results[s])
            gau_m, gau_se = get_stats(gauss_results[s])

            ax.plot(valid_n, bon_m, "k--", linewidth=2, alpha=0.6, label="BoN")
            ax.fill_between(valid_n, bon_m - bon_se, bon_m + bon_se,
                            color="black", alpha=0.1)

            ax.plot(valid_n, gau_m, marker="s", color="#2f855a",
                    linewidth=2, markersize=5, label="PrivBoN-Gaussian")
            ax.fill_between(valid_n, gau_m - gau_se, gau_m + gau_se,
                            color="#2f855a", alpha=0.2)

            ax.plot(valid_n, gum_m, marker="^", color="#dd6b20",
                    linewidth=2, markersize=5, label="PrivBoN-Gumbel")
            ax.fill_between(valid_n, gum_m - gum_se, gum_m + gum_se,
                            color="#dd6b20", alpha=0.2)

            ylabel = r"% Lift over $\pi_{\mathrm{ref}}$"
        else:
            gum_m = get_means(gumbel_results[s])
            gau_m = get_means(gauss_results[s])

            ax.plot(valid_n, bon_m, "k--", linewidth=2.5,
                    label=r"Standard BoN ($\sigma=0$)")
            ax.plot(valid_n, gum_m, marker="o", markersize=6, color="teal",
                    linewidth=2, label="PrivBoN-Gumbel")
            ax.plot(valid_n, gau_m, marker="s", markersize=6, color="salmon",
                    linewidth=2, label="PrivBoN-Gaussian")

            ylabel = r"Expected Proxy Reward $\mathbb{E}[\hat{r}]$"

        ax.set_xscale("log", base=2)
        ax.set_xticks(valid_n[::2])
        ax.set_xticklabels([fr"$2^{{{int(np.log2(n))}}}$" for n in valid_n[::2]])
        ax.set_ylim(y_min, y_max)
        ax.set_title(fr"Noise Level: $\sigma={s}$", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

        if i >= 3:
            ax.set_xlabel("Number of Candidates ($N$)", fontsize=13)
        if i % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        if i == 0:
            ax.legend(fontsize=10, loc="upper left")

    title_metric = "Accuracy Lift" if args.metric == "accuracy_lift" else "Estimated Reward"
    plt.suptitle(
        f"BoN vs Private BoN: {title_metric}\n({args.dataset_label} | {args.rm_label} | Normalized)",
        fontsize=18, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plt.savefig(f"{args.output_basename}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.output_basename}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output_basename}.pdf and {args.output_basename}.png")


if __name__ == "__main__":
    main()
