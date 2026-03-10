"""
Simulate the sigma PID controller across different conditions.
Generates convergence plots and summary statistics as evidence.
"""

import math
import random
from unittest.mock import patch

import matplotlib.pyplot as plt

from pid import SigmaController, DistanceBin


def make_histogram(sigma_true: float, n_bins: int = 40, max_distance: float = 5.0,
                   impressions_per_bin: int = 200, noise: float = 0.0) -> list[DistanceBin]:
    """Generate a synthetic histogram from a Gaussian decay curve with optional noise."""
    bins = []
    for i in range(n_bins):
        d = (i + 0.5) * max_distance / n_bins
        rate = math.exp(-d ** 2 / (2 * sigma_true ** 2))
        if noise > 0:
            rate = max(0, rate + random.gauss(0, noise))
        conversions = int(rate * impressions_per_bin)
        conversions = max(0, min(impressions_per_bin, conversions))
        bins.append(DistanceBin(distance=d, impressions=impressions_per_bin, conversions=conversions))
    return bins


def simulate_convergence(true_sigma: float, target_rate: float, initial_sigma: float,
                         n_steps: int = 300, seed: int = 42, noise: float = 0.02) -> dict:
    """Simulate sigma convergence from an initial guess toward equilibrium."""
    random.seed(seed)
    ctrl = SigmaController(target_rate=target_rate, sigma=initial_sigma, kp=0.3, ki=0.02, kd=0.08)

    sigma_history = [ctrl.sigma]
    boundary_rate_history = []

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(n_steps):
            t += 1.0
            mock_time.monotonic.return_value = t

            histogram = make_histogram(sigma_true=true_sigma, noise=noise)
            boundary_rate = ctrl._boundary_conversion_rate(histogram)
            boundary_rate_history.append(boundary_rate)
            ctrl.update(histogram)
            sigma_history.append(ctrl.sigma)

    # Compute theoretical equilibrium: distance where rate = target_rate
    # rate = exp(-d^2 / (2 * sigma_true^2)) = target_rate
    # d = sigma_true * sqrt(-2 * ln(target_rate))
    if target_rate > 0:
        equilibrium_sigma = true_sigma * math.sqrt(-2 * math.log(target_rate))
    else:
        equilibrium_sigma = float("inf")

    return {
        "true_sigma": true_sigma,
        "target_rate": target_rate,
        "initial_sigma": initial_sigma,
        "equilibrium_sigma": equilibrium_sigma,
        "sigma_history": sigma_history,
        "boundary_rate_history": boundary_rate_history,
        "final_sigma": ctrl.sigma,
    }


def simulate_competitor_entry(true_sigma: float = 1.5, target_rate: float = 0.10,
                              n_steps: int = 400, shock_at: int = 200,
                              seed: int = 42) -> dict:
    """Simulate a competitor entering — the effective Gaussian narrows."""
    random.seed(seed)
    ctrl = SigmaController(target_rate=target_rate, sigma=2.0, kp=0.3, ki=0.02, kd=0.08)

    sigma_history = [ctrl.sigma]
    boundary_rate_history = []

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(n_steps):
            t += 1.0
            mock_time.monotonic.return_value = t

            # After shock, the effective conversion curve narrows
            # (competitor takes the edges)
            if i >= shock_at:
                effective_sigma = true_sigma * 0.6
            else:
                effective_sigma = true_sigma

            histogram = make_histogram(sigma_true=effective_sigma, noise=0.02)
            boundary_rate = ctrl._boundary_conversion_rate(histogram)
            boundary_rate_history.append(boundary_rate)
            ctrl.update(histogram)
            sigma_history.append(ctrl.sigma)

    return {
        "sigma_history": sigma_history,
        "boundary_rate_history": boundary_rate_history,
        "shock_at": shock_at,
        "target_rate": target_rate,
    }


def simulate_initial_estimate(true_sigma: float = 1.5, seed: int = 42) -> dict:
    """Show that estimate_sigma_from_curve gets a reasonable starting point."""
    random.seed(seed)
    ctrl = SigmaController(target_rate=0.10, sigma=5.0)  # bad initial guess

    histogram = make_histogram(sigma_true=true_sigma, n_bins=40, impressions_per_bin=1000)
    estimated = ctrl.estimate_sigma_from_curve(histogram)

    return {
        "true_sigma": true_sigma,
        "bad_initial": 5.0,
        "estimated": estimated,
    }


def plot_convergence(results: list[dict], filename: str = "convergence.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for r in results:
        label = f"true σ={r['true_sigma']}, target={r['target_rate']:.0%}, start={r['initial_sigma']}"
        axes[0].plot(r["sigma_history"], label=label, alpha=0.8)
        axes[0].axhline(y=r["equilibrium_sigma"], color="gray", linestyle=":", alpha=0.3)

    axes[0].set_ylabel("σ")
    axes[0].set_title("Sigma Convergence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for r in results:
        label = f"target={r['target_rate']:.0%}"
        axes[1].plot(r["boundary_rate_history"], label=label, alpha=0.8)
        axes[1].axhline(y=r["target_rate"], color="gray", linestyle="--", alpha=0.3)

    axes[1].set_ylabel("Boundary Conversion Rate")
    axes[1].set_xlabel("Update Steps")
    axes[1].set_title("Conversion Rate at Boundary")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_shock(result: dict, filename: str = "competitor_entry.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(result["sigma_history"], color="steelblue", alpha=0.8)
    axes[0].axvline(x=result["shock_at"], color="red", linestyle="--", alpha=0.5, label="Competitor enters")
    axes[0].set_ylabel("σ")
    axes[0].set_title("Sigma Recovery After Competitor Entry")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result["boundary_rate_history"], color="steelblue", alpha=0.8)
    axes[1].axhline(y=result["target_rate"], color="gray", linestyle="--", alpha=0.5, label="Target")
    axes[1].axvline(x=result["shock_at"], color="red", linestyle="--", alpha=0.5, label="Competitor enters")
    axes[1].set_ylabel("Boundary Conversion Rate")
    axes[1].set_xlabel("Update Steps")
    axes[1].set_title("Conversion Rate Recovery")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def print_summary(results: list[dict]):
    print("\n=== Convergence Summary ===\n")
    print(f"{'True σ':>8} {'Target':>8} {'Start':>8} {'Equil.':>8} {'Final σ':>8} {'|Error|':>8}")
    print("-" * 52)
    for r in results:
        error = abs(r["final_sigma"] - r["equilibrium_sigma"])
        print(f"{r['true_sigma']:>8.2f} {r['target_rate']:>8.0%} {r['initial_sigma']:>8.2f} "
              f"{r['equilibrium_sigma']:>8.2f} {r['final_sigma']:>8.2f} {error:>8.3f}")


def simulate_noisy_histograms(true_sigma: float = 1.5, target_rate: float = 0.10,
                               n_steps: int = 300, seed: int = 42) -> dict:
    """
    Compare convergence under different noise levels.
    Real histograms have sampling noise; this shows the controller is robust to it.
    """
    noise_levels = [0.0, 0.05, 0.10, 0.15]
    all_results = []

    for noise in noise_levels:
        r = simulate_convergence(true_sigma=true_sigma, target_rate=target_rate,
                                 initial_sigma=3.0, n_steps=n_steps, seed=seed, noise=noise)
        r["noise"] = noise
        all_results.append(r)

    return all_results


def simulate_sparse_histograms(true_sigma: float = 1.5, target_rate: float = 0.10,
                                n_steps: int = 300, seed: int = 42) -> dict:
    """
    Simulate with small bin counts that trigger bin suppression (< 11 impressions).
    Shows the controller still works with reduced data after suppression.
    """
    random.seed(seed)

    from pid import SigmaController, validate_histogram
    ctrl = SigmaController(target_rate=target_rate, sigma=3.0, kp=0.3, ki=0.02, kd=0.08)

    sigma_history = [ctrl.sigma]
    bins_before = []
    bins_after = []

    with patch("pid.time") as mock_time:
        t = 0.0
        ctrl._prev_time = t

        for i in range(n_steps):
            t += 1.0
            mock_time.monotonic.return_value = t

            # Mix of well-populated and sparse bins
            histogram = make_histogram(sigma_true=true_sigma, n_bins=40,
                                       impressions_per_bin=random.choice([5, 8, 15, 50, 200]),
                                       noise=0.02)
            safe = validate_histogram(histogram)
            bins_before.append(len(histogram))
            bins_after.append(len(safe))

            ctrl.update(histogram)  # controller calls validate internally
            sigma_history.append(ctrl.sigma)

    return {
        "sigma_history": sigma_history,
        "bins_before": bins_before,
        "bins_after": bins_after,
        "avg_suppressed": sum(b - a for b, a in zip(bins_before, bins_after)) / len(bins_before),
    }


def simulate_multiple_seeds(true_sigma: float = 1.5, target_rate: float = 0.10,
                            n_seeds: int = 20, n_steps: int = 300) -> dict:
    """Run the same scenario across multiple random seeds to show consistency."""
    all_sigma = []
    final_sigmas = []

    for seed in range(n_seeds):
        r = simulate_convergence(true_sigma=true_sigma, target_rate=target_rate,
                                 initial_sigma=3.0, n_steps=n_steps, seed=seed, noise=0.03)
        all_sigma.append(r["sigma_history"])
        final_sigmas.append(r["final_sigma"])

    return {
        "target_rate": target_rate,
        "equilibrium": r["equilibrium_sigma"],
        "all_sigma": all_sigma,
        "final_sigmas": final_sigmas,
        "mean_final": sum(final_sigmas) / len(final_sigmas),
        "n_seeds": n_seeds,
    }


def plot_noise(results: list[dict], filename: str = "noise_robustness.png"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for r in results:
        label = f"noise={r['noise']:.2f}"
        ax.plot(r["sigma_history"], label=label, alpha=0.7)

    ax.axhline(y=results[0]["equilibrium_sigma"], color="gray", linestyle=":", alpha=0.5, label="Equilibrium")
    ax.set_ylabel("σ")
    ax.set_xlabel("Update Steps")
    ax.set_title("Sigma Convergence Under Histogram Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_suppression(result: dict, filename: str = "bin_suppression.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(result["sigma_history"], color="steelblue", alpha=0.8)
    axes[0].set_ylabel("σ")
    axes[0].set_title(f"Sigma With Bin Suppression (avg {result['avg_suppressed']:.1f} bins suppressed/step)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result["bins_before"], alpha=0.5, color="gray", label="Bins before suppression")
    axes[1].plot(result["bins_after"], alpha=0.8, color="steelblue", label="Bins after suppression")
    axes[1].set_ylabel("Bin Count")
    axes[1].set_xlabel("Update Steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_robustness(result: dict, filename: str = "robustness.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for sigma_hist in result["all_sigma"]:
        axes[0].plot(sigma_hist, alpha=0.2, color="steelblue")
    axes[0].axhline(y=result["equilibrium"], color="red", linestyle="--", alpha=0.5, label="Equilibrium")
    axes[0].set_ylabel("σ")
    axes[0].set_xlabel("Update Steps")
    axes[0].set_title(f"Sigma Across {result['n_seeds']} Random Seeds (target={result['target_rate']:.0%})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(result["final_sigmas"], bins=12, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=result["equilibrium"], color="red", linestyle="--", label="Equilibrium")
    axes[1].axvline(x=result["mean_final"], color="orange", linestyle="--",
                    label=f"Mean={result['mean_final']:.2f}")
    axes[1].set_xlabel("Final σ")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Final Sigma Across Seeds")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    # Test convergence from different starting points
    scenarios = [
        {"true_sigma": 1.5, "target_rate": 0.10, "initial_sigma": 0.5},
        {"true_sigma": 1.5, "target_rate": 0.10, "initial_sigma": 4.0},
        {"true_sigma": 1.5, "target_rate": 0.20, "initial_sigma": 3.0},
        {"true_sigma": 2.0, "target_rate": 0.05, "initial_sigma": 1.0},
    ]

    results = [simulate_convergence(**s) for s in scenarios]
    print_summary(results)
    plot_convergence(results)

    # Test initial estimate
    est = simulate_initial_estimate()
    print(f"\nInitial estimate: true σ={est['true_sigma']}, "
          f"bad guess={est['bad_initial']}, estimated={est['estimated']:.3f}")

    # Test competitor entry
    shock = simulate_competitor_entry()
    plot_shock(shock)

    # Test noise robustness
    noise_results = simulate_noisy_histograms()
    plot_noise(noise_results)

    # Test bin suppression
    sparse = simulate_sparse_histograms()
    plot_suppression(sparse)

    # Test across multiple seeds
    robustness = simulate_multiple_seeds()
    plot_robustness(robustness)
    print(f"\nRobustness: mean final σ = {robustness['mean_final']:.3f} "
          f"(equilibrium = {robustness['equilibrium']:.3f}) across {robustness['n_seeds']} seeds")

    print("\nDone.")
