"""Plotting and output utilities for structure search experiments."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_search_comparison(
    residual_history: dict,
    random_histories: list[dict],
    save_path: str,
):
    """Coverage gap vs search step: residual-driven line + random mean±std band."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    steps = range(len(residual_history["coverage_gap"]))

    # Coverage gap
    ax = axes[0]
    ax.plot(steps, residual_history["coverage_gap"], "b-o", label="Residual-driven", linewidth=2)
    if random_histories:
        rand_gaps = np.array([h["coverage_gap"] for h in random_histories])
        # Pad to same length
        max_len = max(len(steps), max(len(h["coverage_gap"]) for h in random_histories))
        padded = np.full((len(random_histories), max_len), np.nan)
        for i, h in enumerate(random_histories):
            padded[i, : len(h["coverage_gap"])] = h["coverage_gap"]
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        r_steps = range(len(mean))
        ax.plot(r_steps, mean, "r--", label="Random (mean)", linewidth=2)
        ax.fill_between(r_steps, mean - std, mean + std, alpha=0.2, color="r")
    ax.set_xlabel("Search step")
    ax.set_ylabel("Coverage gap")
    ax.set_title("Coverage Gap vs Search Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Conditioning penalty
    ax = axes[1]
    ax.plot(steps, residual_history["conditioning_penalty"], "b-o", label="Residual-driven", linewidth=2)
    if random_histories:
        padded_cp = np.full((len(random_histories), max_len), np.nan)
        for i, h in enumerate(random_histories):
            padded_cp[i, : len(h["conditioning_penalty"])] = h["conditioning_penalty"]
        mean_cp = np.nanmean(padded_cp, axis=0)
        std_cp = np.nanstd(padded_cp, axis=0)
        ax.plot(r_steps, mean_cp, "r--", label="Random (mean)", linewidth=2)
        ax.fill_between(r_steps, mean_cp - std_cp, mean_cp + std_cp, alpha=0.2, color="r")
    ax.set_xlabel("Search step")
    ax.set_ylabel("Conditioning penalty")
    ax.set_title("Conditioning Penalty vs Search Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved search comparison plot to {save_path}")


def plot_eigenspectrum(eig_history: list[np.ndarray], save_path: str):
    """Eigenvalue spectrum at each search step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    n_steps = len(eig_history)
    for i, eigs in enumerate(eig_history):
        color = cmap(i / max(n_steps - 1, 1))
        ax.semilogy(range(len(eigs)), eigs, "-", color=color, alpha=0.7, label=f"Step {i}")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Eigenspectrum Evolution During Search")
    if n_steps <= 10:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved eigenspectrum plot to {save_path}")


def plot_training_curves(
    residual_accs: list[float],
    random_accs_list: list[list[float]],
    save_path: str,
):
    """Test accuracy over training epochs for residual-driven vs random."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(residual_accs) + 1)
    ax.plot(epochs, residual_accs, "b-o", label="Residual-driven", linewidth=2)
    if random_accs_list:
        arr = np.array(random_accs_list)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(epochs, mean, "r--", label="Random (mean)", linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color="r")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Training Accuracy of Discovered Architectures")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_score_vs_step(
    residual_history: dict,
    random_histories: list[dict],
    save_path: str,
):
    """Total score (phi_T) vs search step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = range(len(residual_history["phi_T"]))
    ax.plot(steps, residual_history["phi_T"], "b-o", label="Residual-driven", linewidth=2)
    if random_histories:
        max_len = max(len(h["phi_T"]) for h in random_histories)
        padded = np.full((len(random_histories), max_len), np.nan)
        for i, h in enumerate(random_histories):
            padded[i, : len(h["phi_T"])] = h["phi_T"]
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        r_steps = range(len(mean))
        ax.plot(r_steps, mean, "r--", label="Random (mean)", linewidth=2)
        ax.fill_between(r_steps, mean - std, mean + std, alpha=0.2, color="r")
    ax.set_xlabel("Search step")
    ax.set_ylabel("φ_T (profiled energy)")
    ax.set_title("Profiled Energy vs Search Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved score plot to {save_path}")


def plot_energy_gap(
    residual_history: dict,
    random_histories: list[dict],
    save_path: str,
):
    """Energy gap (discrimination) vs search step."""
    if "energy_gap" not in residual_history or not residual_history["energy_gap"]:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = range(len(residual_history["energy_gap"]))
    ax.plot(steps, residual_history["energy_gap"], "b-o", label="Residual-driven", linewidth=2)
    if random_histories:
        max_len = max(len(h.get("energy_gap", [])) for h in random_histories)
        if max_len > 0:
            padded = np.full((len(random_histories), max_len), np.nan)
            for i, h in enumerate(random_histories):
                eg = h.get("energy_gap", [])
                padded[i, : len(eg)] = eg
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            r_steps = range(len(mean))
            ax.plot(r_steps, mean, "r--", label="Random (mean)", linewidth=2)
            ax.fill_between(r_steps, mean - std, mean + std, alpha=0.2, color="r")
    ax.set_xlabel("Search step")
    ax.set_ylabel("Energy gap (E_wrong - E_true)")
    ax.set_title("Energy Gap (Discriminative Power) vs Search Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved energy gap plot to {save_path}")


def plot_transform_types(
    residual_edges: list[str],
    random_edges_list: list[list[str]],
    save_path: str,
):
    """Bar chart of transform type counts: residual-driven vs random."""
    import re

    def extract_types(edges):
        types = {}
        for e in edges:
            m = re.search(r"\[(\w+)\]", e)
            t = m.group(1) if m else "Linear"
            types[t] = types.get(t, 0) + 1
        return types

    res_types = extract_types(residual_edges)
    all_types = set(res_types.keys())
    rand_type_counts = []
    for edges in random_edges_list:
        rt = extract_types(edges)
        all_types.update(rt.keys())
        rand_type_counts.append(rt)

    types_sorted = sorted(all_types)
    x = np.arange(len(types_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    res_vals = [res_types.get(t, 0) for t in types_sorted]
    ax.bar(x - width / 2, res_vals, width, label="Residual-driven", color="steelblue")
    if rand_type_counts:
        rand_arr = np.array([[rt.get(t, 0) for t in types_sorted] for rt in rand_type_counts])
        rand_mean = rand_arr.mean(axis=0)
        rand_std = rand_arr.std(axis=0)
        ax.bar(x + width / 2, rand_mean, width, yerr=rand_std, label="Random (mean)", color="salmon", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(types_sorted)
    ax.set_xlabel("Transform type")
    ax.set_ylabel("Count")
    ax.set_title("Transform Types Selected by Search")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved transform types plot to {save_path}")


def save_results(results: dict, output_dir: str):
    """Save search results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"Saved results to {path}")
