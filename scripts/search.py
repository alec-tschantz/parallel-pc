"""Structure learning demo: A-optimal backward elimination.

J(S) = tr(H_S⁻¹ Σ_task)   — lower = better structure

Builds a graph with known structure and runs backward elimination.
The algorithm should prune noise edges (Δ≈0) while retaining essential
and skip edges that provide task-relevant precision.

Graph topology:
  x ──→ h0 ──→ h1 ──→ h2 ──→ h3 ──→ y     (essential chain)
         ╰──────────→ h2                     (skip: h0→h2)
                h1 ──────────→ h3            (skip: h1→h3)
         h0 ──→ d0                           (dead-end)
                h1 ──→ d1                    (dead-end)
                       h2 ──→ d2             (dead-end)
         h0 ──→ h1 [dup]                     (duplicate)
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import ppc


# ── Graph construction ───────────────────────────────────────────────

EDGE_CATEGORIES = {}  # edge_idx → category string


def build_graph(key, dim_x=6, dim_h=4, dim_y=6, dim_d=4):
    """Build test graph with essential, skip, dead-end, and duplicate edges."""
    variables = [
        ppc.Variable("x", (dim_x,)),
        ppc.Variable("h0", (dim_h,)),
        ppc.Variable("h1", (dim_h,)),
        ppc.Variable("h2", (dim_h,)),
        ppc.Variable("h3", (dim_h,)),
        ppc.Variable("y", (dim_y,)),
        ppc.Variable("d0", (dim_d,)),
        ppc.Variable("d1", (dim_d,)),
        ppc.Variable("d2", (dim_d,)),
    ]

    transforms = []
    energies = []

    def add(name, src, tgt, in_dim, out_dim, category):
        nonlocal key
        key, k = jax.random.split(key)
        idx = len(transforms)
        transforms.append(
            ppc.Transform(name, eqx.nn.Linear(in_dim, out_dim, key=k), src=src, tgt=tgt)
        )
        energies.append(ppc.Energy(ppc.mse_energy, args=[name, tgt]))
        EDGE_CATEGORIES[idx] = category

    # Essential chain: x → h0 → h1 → h2 → h3 → y
    add("x→h0", "x", "h0", dim_x, dim_h, "essential")
    add("h0→h1", "h0", "h1", dim_h, dim_h, "essential")
    add("h1→h2", "h1", "h2", dim_h, dim_h, "essential")
    add("h2→h3", "h2", "h3", dim_h, dim_h, "essential")
    add("h3→y", "h3", "y", dim_h, dim_y, "essential")

    # Skip connections (bypass depth)
    add("h0→h2[skip]", "h0", "h2", dim_h, dim_h, "skip")
    add("h1→h3[skip]", "h1", "h3", dim_h, dim_h, "skip")

    # Dead-end branches (task-irrelevant)
    add("h0→d0", "h0", "d0", dim_h, dim_d, "dead-end")
    add("h1→d1", "h1", "d1", dim_h, dim_d, "dead-end")
    add("h2→d2", "h2", "d2", dim_h, dim_d, "dead-end")

    # Duplicate (redundant copy of chain link)
    add("h0→h1[dup]", "h0", "h1", dim_h, dim_h, "duplicate")

    graph = ppc.Graph(variables=variables, transforms=transforms, energies=energies)
    return graph


# ── Plotting ─────────────────────────────────────────────────────────

CAT_COLORS = {
    "essential": "#2E86AB",
    "skip": "#A23B72",
    "dead-end": "#F18F01",
    "duplicate": "#C73E1D",
}
CAT_MARKERS = {
    "essential": "o",
    "skip": "s",
    "dead-end": "^",
    "duplicate": "D",
}


def plot_results(diag, graph, output_dir):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        r"A-optimal backward elimination:  $J(S) = \mathrm{tr}(H_S^{-1}\,\Sigma_{\mathrm{task}})$"
        "\nlower score = better structure",
        fontsize=14,
        fontweight="bold",
    )

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # ── Panel 1: Pruning trajectory ──────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    history = diag["history"]
    steps = list(range(len(history)))
    scores = [h["score"] for h in history]

    # Background shading for phases
    dead_end_steps = [
        i
        for i, h in enumerate(history[1:], 1)
        if EDGE_CATEGORIES.get(h.get("removed", -1)) == "dead-end"
    ]
    if dead_end_steps:
        ax.axvspan(
            -0.5,
            max(dead_end_steps) + 0.5,
            alpha=0.06,
            color=CAT_COLORS["dead-end"],
            label="_nolegend_",
        )
        ax.text(
            np.mean(dead_end_steps),  # type: ignore
            max(scores) * 0.97,
            "dead-ends",
            ha="center",
            fontsize=8,
            color=CAT_COLORS["dead-end"],
            alpha=0.8,
        )

    # Color each step by the category of the edge that was removed
    for i, h in enumerate(history):
        if i == 0:
            ax.plot(i, h["score"], "ko", ms=9, zorder=5)
            continue
        cat = EDGE_CATEGORIES.get(h["removed"], "essential")
        ax.plot(
            i,
            h["score"],
            marker=CAT_MARKERS[cat],
            color=CAT_COLORS[cat],
            ms=9,
            zorder=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
        )
    ax.plot(steps, scores, "k-", alpha=0.3, linewidth=1.5)
    ax.set_xlabel("Edges removed", fontsize=11)
    ax.set_ylabel("Score (lower = better)", fontsize=11)
    ax.set_title("Pruning trajectory", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)

    # Annotate every step
    prev_y = -1e9
    for i, h in enumerate(history[1:], 1):
        name = h.get("removed_name", "")
        y = h["score"]
        # Alternate label placement to avoid overlap
        offset_y = 12 if (y - prev_y) < (max(scores) - min(scores)) * 0.05 else -14
        ax.annotate(
            name,
            (i, y),
            textcoords="offset points",
            xytext=(6, offset_y),
            fontsize=7,
            alpha=0.85,
            arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5),
        )
        prev_y = y

    # ── Panel 2: Marginal contributions (bar chart) ────────────────
    ax = fig.add_subplot(gs[0, 2])
    # Compute marginals for all internal edges at full graph
    state = ppc.init(
        graph,
        {
            "x": jax.random.normal(
                jax.random.PRNGKey(99), (32, graph.layout.shapes["x"][0])
            ),
            "y": jax.random.normal(
                jax.random.PRNGKey(100), (32, graph.layout.shapes["y"][0])
            ),
        },
        key=jax.random.PRNGKey(101),
    )
    all_J, all_r = ppc.precompute_edge_data(graph, state)
    edges = ppc.classify_edges(graph, state)
    Sigma_task = ppc.task_covariance(all_J, all_r, edges["boundary"])

    all_edges = list(range(len(graph.transforms)))
    full_score = ppc.score_edge_set(all_J, all_edges, Sigma_task)["score"]

    marginals = {}
    for e in edges["internal"]:
        reduced = [i for i in all_edges if i != e]
        s = ppc.score_edge_set(all_J, reduced, Sigma_task)["score"]
        marginals[e] = s - full_score  # ≥ 0

    # Sort by marginal contribution
    sorted_edges = sorted(marginals.keys(), key=lambda e: marginals[e])
    names = [graph.transforms[e].id for e in sorted_edges]
    vals = [marginals[e] for e in sorted_edges]
    colors = [CAT_COLORS[EDGE_CATEGORIES[e]] for e in sorted_edges]

    bars = ax.barh(
        range(len(names)), vals, color=colors, edgecolor="k", linewidth=0.5, alpha=0.85
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Marginal Δ (removal cost)", fontsize=10)
    ax.set_title(
        "Marginal contributions\n(at full graph)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.15, axis="x")

    # Add value labels
    max_val = max(vals) if vals else 1
    for i, v in enumerate(vals):
        label = f"{v:.4f}" if v > 0.001 else f"{v:.1e}"
        ax.text(v + max_val * 0.03, i, label, va="center", fontsize=8)

    # ── Panel 3: Eigenspectrum ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])

    full_result = ppc.score_edge_set(all_J, all_edges, Sigma_task)
    eigs_full = np.sort(np.array(full_result["eigenvalues"]))[::-1]

    kept_no_dead = [e for e in all_edges if EDGE_CATEGORIES.get(e) != "dead-end"]
    result_no_dead = ppc.score_edge_set(all_J, kept_no_dead, Sigma_task)
    eigs_no_dead = np.sort(np.array(result_no_dead["eigenvalues"]))[::-1]

    result_boundary = ppc.score_edge_set(all_J, edges["boundary"], Sigma_task)
    eigs_boundary = np.sort(np.array(result_boundary["eigenvalues"]))[::-1]

    ax.semilogy(
        eigs_full, "o-", ms=5, label="Full graph", color="#2E86AB", linewidth=1.5
    )
    ax.semilogy(
        eigs_no_dead, "s-", ms=5, label="No dead-ends", color="#A23B72", linewidth=1.5
    )
    ax.semilogy(
        eigs_boundary, "^-", ms=5, label="Boundary only", color="#F18F01", linewidth=1.5
    )
    ax.axhline(
        1e-4, color="gray", ls="--", alpha=0.5, linewidth=1, label="ε = 1e-4 (gate)"
    )
    ax.set_xlabel("Eigenvalue index", fontsize=10)
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=10)
    ax.set_title("Hessian eigenspectrum", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)

    # ── Panel 4: Score decomposition σ_i / (λ_i + ε) ────────────────
    ax = fig.add_subplot(gs[1, 1])

    # Eigendecomposition of H_S
    H = 1e-4 * jnp.eye(eigs_full.shape[0])
    for e in all_edges:
        H = H + all_J[e].T @ all_J[e]
    eigvals, eigvecs = jnp.linalg.eigh(H)
    sigma_i = jnp.diag(eigvecs.T @ Sigma_task @ eigvecs)
    contrib = sigma_i / eigvals  # σ_i / (λ_i + ε)

    # Sort by contribution (largest first)
    order = np.argsort(-np.array(contrib))
    contrib_sorted = np.array(contrib)[order]
    ax.bar(
        range(len(contrib_sorted)),
        contrib_sorted,
        color="#2E86AB",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("Direction index (sorted)", fontsize=10)
    ax.set_ylabel(r"$\sigma_i / (\lambda_i + \varepsilon)$", fontsize=11)
    ax.set_title(
        r"Score decomposition: $J = \sum_i \sigma_i/(\lambda_i + \varepsilon)$",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.15)

    # Annotate: how many directions contribute most of the score
    cumsum = np.cumsum(contrib_sorted) / np.sum(contrib_sorted)
    n90 = np.searchsorted(cumsum, 0.9) + 1
    ax.axvline(n90 - 0.5, color="red", ls="--", alpha=0.5, linewidth=1)
    ax.text(
        n90 + 0.5,
        max(contrib_sorted) * 0.9,
        f"90% of score\nin {n90}/{len(contrib_sorted)} dirs",
        fontsize=8,
        color="red",
        alpha=0.8,
    )

    # ── Panel 5: Cumulative score as edges are removed ───────────────
    ax = fig.add_subplot(gs[1, 2])
    deltas = []
    cats_ordered = []
    for h in history[1:]:
        d = h.get("delta", 0)
        deltas.append(d)
        cats_ordered.append(EDGE_CATEGORIES.get(h.get("removed", -1), ""))

    if deltas:
        x_pos = range(len(deltas))
        bar_colors = [CAT_COLORS.get(c, "gray") for c in cats_ordered]
        ax.bar(
            x_pos, deltas, color=bar_colors, edgecolor="k", linewidth=0.5, alpha=0.85
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [h.get("removed_name", "") for h in history[1:]],
            rotation=45,
            ha="right",
            fontsize=7,
            fontfamily="monospace",
        )
        ax.set_ylabel("Δ (score increase)", fontsize=10)
        ax.set_title(
            "Cost of each removal\n(in elimination order)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_yscale("symlog", linthresh=0.001)
        ax.grid(True, alpha=0.15)

    # Category legend
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=CAT_COLORS[c], edgecolor="k", linewidth=0.5, label=c)
        for c in ["essential", "skip", "dead-end", "duplicate"]
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=True,
        edgecolor="gray",
        bbox_to_anchor=(0.5, -0.01),
    )
    path = os.path.join(output_dir, "search.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Main ─────────────────────────────────────────────────────────────


def main():
    output_dir = "outputs/search"
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(0)
    graph = build_graph(key)

    n_edges = len(graph.transforms)
    n_by_cat = {}
    for cat in EDGE_CATEGORIES.values():
        n_by_cat[cat] = n_by_cat.get(cat, 0) + 1

    print("=" * 64)
    print("  A-optimal structure learning")
    print("  J(S) = tr(H_S⁻¹ Σ_task)   lower = better")
    print("=" * 64)
    print(f"\n  Graph: {n_edges} edges")
    for cat, n in n_by_cat.items():
        print(f"    {cat:12s}: {n}")
    print(f"\n  Edges:")
    for i, t in enumerate(graph.transforms):
        cat = EDGE_CATEGORIES[i]
        print(f"    [{i:2d}] {t.id:20s}  {cat}")

    # Probe batch
    batch = 64
    key, k1, k2 = jax.random.split(key, 3)
    dim_x = graph.layout.shapes["x"][0]
    dim_y = graph.layout.shapes["y"][0]
    clamps = {
        "x": jax.random.normal(k1, (batch, dim_x)),
        "y": jax.random.normal(k2, (batch, dim_y)),
    }

    # Run backward elimination (large delta → full trajectory)
    print(f"\n  Running backward elimination...")
    cfg = ppc.SearchConfig(eps=1e-4, delta=1e6)
    key, k = jax.random.split(key)
    reduced, diag = ppc.reduce(graph, clamps, cfg, k)

    # Pretty output
    print(f"\n  {'─'*62}")
    print(f"  {'Step':>4}  {'Removed':<22} {'Category':<12} {'Score':>10}  {'Δ':>10}")
    print(f"  {'─'*62}")
    for i, h in enumerate(diag["history"]):
        name = h.get("removed_name", "full graph")
        cat = EDGE_CATEGORIES.get(h.get("removed", -1), "")
        d = h.get("delta")
        d_str = f"{d:+.6f}" if d is not None else ""
        print(f"  {i:4d}  {name:<22} {cat:<12} {h['score']:10.6f}  {d_str:>10}")
    print(f"  {'─'*62}")

    # Summary
    pruned = [graph.transforms[i].id for i in diag["pruned_order"]]
    pruned_cats = [EDGE_CATEGORIES[i] for i in diag["pruned_order"]]
    dead_end_positions = [i for i, c in enumerate(pruned_cats) if c == "dead-end"]
    dup_positions = [i for i, c in enumerate(pruned_cats) if c == "duplicate"]
    essential_positions = [i for i, c in enumerate(pruned_cats) if c == "essential"]
    skip_positions = [i for i, c in enumerate(pruned_cats) if c == "skip"]

    print(f"\n  Pruning order by category:")
    print(f"    Dead-ends  pruned at steps: {dead_end_positions}")
    print(f"    Duplicates pruned at steps: {dup_positions}")
    print(f"    Essential  pruned at steps: {essential_positions}")
    print(f"    Skip       pruned at steps: {skip_positions}")

    # Verify expected behaviour
    print(f"\n  Checks:")

    # Monotonicity
    scores = [h["score"] for h in diag["history"]]
    mono = all(scores[i] >= scores[i - 1] - 1e-8 for i in range(1, len(scores)))
    print(f"    {'✓' if mono else '✗'} Score monotonically non-decreasing")

    # Dead-ends pruned first
    if dead_end_positions and essential_positions:
        dead_before_ess = max(dead_end_positions) < min(essential_positions)
        print(
            f"    {'✓' if dead_before_ess else '○'} Dead-ends pruned before essential edges"
        )

    # Dead-end deltas near zero
    dead_deltas = [diag["history"][i + 1]["delta"] for i in dead_end_positions]
    if dead_deltas:
        max_dead_delta = max(dead_deltas)
        print(
            f"    {'✓' if max_dead_delta < 0.01 else '○'} Dead-end Δ max = {max_dead_delta:.6f} (expect ≈ 0)"
        )

    # Plot
    plot_path = plot_results(diag, graph, output_dir)
    print(f"\n  Plot saved to {plot_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
