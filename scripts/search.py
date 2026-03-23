"""Synthetic validation of exact backward elimination.

Generates random graphs with planted structure, runs backward elimination,
checks whether the algorithm correctly identifies essential edges.
All transforms are Linear (GN approximation is exact).
"""

import json
import os
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro

import ppc


@dataclass
class Cfg:
    seed: int = 0
    n_chain: int = 3
    n_dead: int = 3
    n_noise_dead: int = 8
    n_skip: int = 2
    dim: int = 16
    dim_out: int = 8
    batch_size: int = 128
    noise_std: float = 0.1
    n_trials: int = 10
    search: ppc.SearchConfig = field(default_factory=ppc.SearchConfig)
    output_dir: str = "outputs/search"


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def generate_trial(cfg: Cfg, key: jax.Array) -> dict:
    """Random graph with planted essential chain + noise dead-end edges."""
    keys = jax.random.split(key, 100)
    ki = 0

    variables = [ppc.Variable("x", (cfg.dim,)), ppc.Variable("y", (cfg.dim_out,))]
    for i in range(cfg.n_chain):
        variables.append(ppc.Variable(f"h_{i}", (cfg.dim,)))
    for i in range(cfg.n_dead):
        variables.append(ppc.Variable(f"d_{i}", (cfg.dim,)))

    transforms, energies, names, categories = [], [], [], {}

    def add(name, src, tgt, in_dim, out_dim, cat):
        nonlocal ki
        idx = len(transforms)
        tid = f"t_{idx}_{name}"
        transforms.append(ppc.Transform(tid, eqx.nn.Linear(in_dim, out_dim, key=keys[ki]), src=src, tgt=tgt))
        energies.append(ppc.Energy(ppc.mse_energy, args=[tid, tgt]))
        names.append(name)
        categories[idx] = cat
        ki += 1

    # Essential chain: x → h_0 → ... → h_{n-1} → y
    add("x->h_0", "x", "h_0", cfg.dim, cfg.dim, "essential")
    for i in range(cfg.n_chain - 1):
        add(f"h_{i}->h_{i+1}", f"h_{i}", f"h_{i+1}", cfg.dim, cfg.dim, "essential")
    add(f"h_{cfg.n_chain-1}->y", f"h_{cfg.n_chain-1}", "y", cfg.dim, cfg.dim_out, "essential")

    # Skip edges (chain shortcuts)
    if cfg.n_chain >= 3:
        possible = [(i, j) for i in range(cfg.n_chain) for j in range(i + 2, cfg.n_chain)]
        rng = np.random.RandomState(int(keys[ki][0]))
        ki += 1
        rng.shuffle(possible)
        for i, j in possible[:cfg.n_skip]:
            add(f"h_{i}->h_{j}[skip]", f"h_{i}", f"h_{j}", cfg.dim, cfg.dim, "skip")

    # Noise edges (dead-end connections)
    noise_opts = []
    for hi in range(cfg.n_chain):
        for di in range(cfg.n_dead):
            noise_opts.append((f"h_{hi}", f"d_{di}", f"h_{hi}->d_{di}"))
            noise_opts.append((f"d_{di}", f"h_{hi}", f"d_{di}->h_{hi}"))
    for di in range(cfg.n_dead):
        for dj in range(di + 1, cfg.n_dead):
            noise_opts.append((f"d_{di}", f"d_{dj}", f"d_{di}->d_{dj}"))
    rng2 = np.random.RandomState(int(keys[ki][0]))
    ki += 1
    rng2.shuffle(noise_opts)
    for src, tgt, name in noise_opts[:cfg.n_noise_dead]:
        add(name, src, tgt, cfg.dim, cfg.dim, "noise")

    graph = ppc.Graph(variables=variables, transforms=transforms, energies=energies)

    # Independent synthetic data
    x = jax.random.normal(keys[ki], (cfg.batch_size, cfg.dim)); ki += 1
    W = jax.random.normal(keys[ki], (cfg.dim_out, cfg.dim)) * 0.5; ki += 1
    noise = cfg.noise_std * jax.random.normal(keys[ki], (cfg.batch_size, cfg.dim_out)); ki += 1
    y = x @ W.T + noise

    return {
        "graph": graph, "clamps": {"x": x, "y": y},
        "categories": categories, "names": names,
    }


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def auroc(scores: dict[int, float], pos: set[int], neg: set[int]) -> float:
    ps = [scores[i] for i in pos if i in scores]
    ns = [scores[i] for i in neg if i in scores]
    if not ps or not ns:
        return float("nan")
    c = sum(1 if p > n else 0.5 if p == n else 0 for p in ps for n in ns)
    return c / (len(ps) * len(ns))


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_trial(cfg: Cfg, key: jax.Array, num: int) -> dict:
    trial = generate_trial(cfg, key)
    graph, clamps = trial["graph"], trial["clamps"]
    cats, names = trial["categories"], trial["names"]

    n_ess = sum(1 for c in cats.values() if c == "essential")
    n_skip = sum(1 for c in cats.values() if c == "skip")
    n_noise = sum(1 for c in cats.values() if c == "noise")
    print(f"\n  Trial {num}: {len(graph.transforms)} edges "
          f"({n_ess} essential, {n_skip} skip, {n_noise} noise)")

    reduced, diag = ppc.reduce(graph, clamps, cfg.search, key)

    # Which edges survived?
    surviving_ids = {t.id for t in reduced.transforms}
    surviving_idx = {i for i, t in enumerate(graph.transforms) if t.id in surviving_ids}

    # Classify by internal/boundary (just need a state for free_mask)
    state = ppc.init(graph, clamps, key=key)
    edge_class = ppc.classify_edges(graph, state)
    internal = set(edge_class["internal"])

    ess_int = {i for i, c in cats.items() if c == "essential"} & internal
    skip_int = {i for i, c in cats.items() if c == "skip"} & internal
    noise_int = {i for i, c in cats.items() if c == "noise"} & internal
    useful_int = ess_int | skip_int

    # Removal scores: edges removed early had LOW score-without (removing helped)
    # Edges that survived had HIGH score-without (removing would hurt)
    # For AUROC: useful edges should have HIGHER score-without than noise
    pruned = diag["pruned_order"]
    removal_rank = {}  # higher = removed later = more important
    for rank, idx in enumerate(pruned):
        removal_rank[idx] = rank
    # Surviving edges get max rank
    max_rank = len(pruned)
    for i in internal:
        if i not in removal_rank:
            removal_rank[i] = max_rank

    auc = auroc(removal_rank, useful_int, noise_int)

    # Recovery
    ess_survived = ess_int & surviving_idx
    recovery = len(ess_survived) / len(ess_int) if ess_int else 1.0

    # Noise pruned before essential
    noise_first = 0
    for idx in pruned:
        if idx in ess_int:
            break
        if idx in noise_int:
            noise_first += 1

    # Score trajectory
    scores = [h["score"] for h in diag["history"]]

    print(f"    AUROC={auc:.3f}  recovery={recovery:.0%}  "
          f"noise_first={noise_first}/{len(noise_int)}  "
          f"pruned={len(pruned)}/{len(internal)}")

    return {
        "auroc": auc, "recovery": recovery,
        "noise_first": noise_first, "n_noise_int": len(noise_int),
        "n_ess_int": len(ess_int), "n_skip_int": len(skip_int),
        "scores": scores, "pruned_order": pruned,
        "categories": cats, "names": names,
        "removal_rank": removal_rank, "internal": internal,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(results: list[dict], cfg: Cfg, output_dir: str):
    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    recoveries = [r["recovery"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. AUROC
    ax = axes[0]
    ax.boxplot(aurocs, positions=[1], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    for a in aurocs:
        ax.scatter(1 + np.random.uniform(-0.12, 0.12), a, color="navy", s=20, alpha=0.6)
    ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="Chance")
    ax.set_ylabel("AUROC")
    ax.set_title(f"AUROC={np.mean(aurocs):.3f}±{np.std(aurocs):.3f}")
    ax.set_xticks([])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Recovery
    ax = axes[1]
    ax.bar([1], [np.mean(recoveries)], color="seagreen", alpha=0.7, width=0.4)
    ax.errorbar([1], [np.mean(recoveries)], yerr=[np.std(recoveries)], color="k", capsize=5)
    ax.set_ylabel("Recovery")
    ax.set_title(f"Recovery={np.mean(recoveries):.0%}±{np.std(recoveries):.0%}")
    ax.set_xticks([])
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Score during elimination (overlay trials)
    ax = axes[2]
    for r in results:
        ax.plot(range(len(r["scores"])), r["scores"], "o-", alpha=0.4, ms=3)
    ax.set_xlabel("Elimination step")
    ax.set_ylabel("Boundary score")
    ax.set_title("Score during backward elimination")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"chain={cfg.n_chain} dead={cfg.n_dead} noise={cfg.n_noise_dead} "
        f"skip={cfg.n_skip} dim={cfg.dim} eps={cfg.search.eps}",
        fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Per-trial: removal order colored by category (first 3 only)
    for ti, r in enumerate(results[:3]):
        fig, ax = plt.subplots(figsize=(max(6, len(r["internal"]) * 0.4), 4))
        internal_sorted = sorted(r["internal"], key=lambda i: r["removal_rank"].get(i, 999))
        colors = [{"essential": "steelblue", "skip": "seagreen", "noise": "salmon"}.get(
            r["categories"].get(i, "?"), "gray") for i in internal_sorted]
        labels = [r["names"][i] if i < len(r["names"]) else str(i) for i in internal_sorted]
        ranks = [r["removal_rank"].get(i, len(r["pruned_order"])) for i in internal_sorted]
        ax.barh(range(len(internal_sorted)), ranks, color=colors)
        ax.set_yticks(range(len(internal_sorted)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Removal rank (higher = kept longer)")
        ax.set_title(f"Trial {ti}: AUROC={r['auroc']:.3f} recovery={r['recovery']:.0%}")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="steelblue", label="Essential"),
                           Patch(color="seagreen", label="Skip"),
                           Patch(color="salmon", label="Noise")], loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"trial_{ti}.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    print(f"chain={cfg.n_chain} dead={cfg.n_dead} noise={cfg.n_noise_dead} "
          f"skip={cfg.n_skip} dim={cfg.dim} eps={cfg.search.eps} trials={cfg.n_trials}")

    results = []
    for t in range(cfg.n_trials):
        key, tk = jax.random.split(key)
        results.append(run_trial(cfg, tk, t))

    aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    recoveries = [r["recovery"] for r in results]

    print(f"\n{'='*50}")
    print(f"AUROC:    {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    print(f"Recovery: {np.mean(recoveries):.0%} ± {np.std(recoveries):.0%}")
    print(f"{'='*50}")

    plot_results(results, cfg, cfg.output_dir)

    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump({
            "config": {k: str(v) for k, v in vars(cfg).items()},
            "summary": {"auroc": float(np.mean(aurocs)), "recovery": float(np.mean(recoveries))},
            "trials": [{"auroc": r["auroc"], "recovery": r["recovery"],
                         "noise_first": r["noise_first"]} for r in results],
        }, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))

    print(f"Saved to {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(Cfg)
    main(cfg)
