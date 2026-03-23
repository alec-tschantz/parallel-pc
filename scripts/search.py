import json
import os
import time
from dataclasses import dataclass, field
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro

import ppc


@dataclass
class Cfg:
    seed: int = 0
    n_chain: int = 3
    n_dead: int = 2
    n_noise: int = 6
    n_skip: int = 1
    dim: int = 16
    dim_hidden: int = 4  # hidden dim (bottleneck)
    dim_out: int = 8
    batch_size: int = 128
    noise_std: float = 0.1
    # Training
    train_size: int = 2048
    test_size: int = 512
    train_epochs: int = 30
    train_lr: float = 1e-4
    infer_lr: float = 0.05
    infer_iters: int = 20
    train_batch_size: int = 64
    # Subgraph sampling
    n_random_subsets: int = 30
    # Search
    search: ppc.SearchConfig = field(default_factory=ppc.SearchConfig)
    output_dir: str = "outputs/search"


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def generate_task(cfg: Cfg, key: jax.Array) -> dict:
    keys = jax.random.split(key, 100)
    ki = 0

    dh = cfg.dim_hidden
    variables = [ppc.Variable("x", (cfg.dim,)), ppc.Variable("y", (cfg.dim_out,))]
    for i in range(cfg.n_chain):
        variables.append(ppc.Variable(f"h_{i}", (dh,)))
    for i in range(cfg.n_dead):
        variables.append(ppc.Variable(f"d_{i}", (dh,)))

    transforms, energies, categories = [], [], {}

    def add(name, src, tgt, in_dim, out_dim, cat):
        nonlocal ki
        idx = len(transforms)
        tid = f"t_{idx}_{name}"
        transforms.append(
            ppc.Transform(
                tid, eqx.nn.Linear(in_dim, out_dim, key=keys[ki]), src=src, tgt=tgt
            )
        )
        energies.append(ppc.Energy(ppc.mse_energy, args=[tid, tgt]))
        categories[idx] = cat
        ki += 1

    # Essential chain: x → h_0 → ... → h_{n-1} → y
    add("x->h_0", "x", "h_0", cfg.dim, dh, "essential")
    for i in range(cfg.n_chain - 1):
        add(f"h_{i}->h_{i+1}", f"h_{i}", f"h_{i+1}", dh, dh, "essential")
    add(
        f"h_{cfg.n_chain-1}->y", f"h_{cfg.n_chain-1}", "y", dh, cfg.dim_out, "essential"
    )

    # Skip edges
    if cfg.n_chain >= 3:
        possible = [
            (i, j) for i in range(cfg.n_chain) for j in range(i + 2, cfg.n_chain)
        ]
        rng = np.random.RandomState(int(keys[ki][0]))
        ki += 1
        rng.shuffle(possible)
        for i, j in possible[: cfg.n_skip]:
            add(f"h_{i}->h_{j}[skip]", f"h_{i}", f"h_{j}", dh, dh, "skip")

    # Noise dead-end edges
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
    for src, tgt, name in noise_opts[: cfg.n_noise]:
        add(name, src, tgt, dh, dh, "noise")

    graph = ppc.Graph(variables=variables, transforms=transforms, energies=energies)

    # Data: y = Wx + noise (independent of graph params)
    W_true = jax.random.normal(keys[ki], (cfg.dim_out, cfg.dim)) * 0.5
    ki += 1
    total = cfg.train_size + cfg.test_size
    x_all = jax.random.normal(keys[ki], (total, cfg.dim))
    ki += 1
    noise = cfg.noise_std * jax.random.normal(keys[ki], (total, cfg.dim_out))
    ki += 1
    y_all = x_all @ W_true.T + noise

    return {
        "graph": graph,
        "categories": categories,
        "x_tr": x_all[: cfg.train_size],
        "y_tr": y_all[: cfg.train_size],
        "x_te": x_all[cfg.train_size :],
        "y_te": y_all[cfg.train_size :],
    }


# ---------------------------------------------------------------------------
# Build subgraph from index set
# ---------------------------------------------------------------------------


def build_subgraph(graph: ppc.Graph, keep: list[int]) -> ppc.Graph:
    kept_t = [graph.transforms[i] for i in keep]
    kept_tids = {graph.transforms[i].id for i in keep}
    kept_e = [
        e
        for e in graph.energies
        if all(
            a in kept_tids or not any(a == t.id for t in graph.transforms)
            for a in e.args
        )
    ]
    return ppc.Graph(
        variables=list(graph.variables), transforms=kept_t, energies=kept_e
    )


# ---------------------------------------------------------------------------
# Train a subgraph, return test MSE
# ---------------------------------------------------------------------------


def train_subgraph(
    subgraph: ppc.Graph,
    cfg: Cfg,
    x_tr: jax.Array,
    y_tr: jax.Array,
    x_te: jax.Array,
    y_te: jax.Array,
    key: jax.Array,
) -> float:
    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(subgraph, eqx.is_array))
    bs = cfg.train_batch_size

    @eqx.filter_jit
    def train_step(g, opt_s, x_batch, y_batch, k):
        state = ppc.init(g, {"x": x_batch, "y": y_batch}, key=k)
        state = ppc.infer(g, state, optimizer=infer_opt, iters=cfg.infer_iters)
        loss = ppc.energy(g, state)
        grads = ppc.param_grad(g, state)
        updates, opt_s = train_opt.update(
            eqx.filter(grads, eqx.is_array), opt_s, eqx.filter(g, eqx.is_array)
        )
        g = eqx.apply_updates(g, updates)
        return g, opt_s, loss

    @eqx.filter_jit
    def eval_mse(g, x_batch, y_batch, k):
        state = ppc.init(g, {"x": x_batch}, key=k)
        state = ppc.infer(g, state, optimizer=infer_opt, iters=cfg.infer_iters * 2)
        y_pred = ppc.variable(g, state, "y")
        return jnp.mean((y_pred - y_batch) ** 2)

    for epoch in range(cfg.train_epochs):
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, x_tr.shape[0])
        for i in range(0, x_tr.shape[0] - bs + 1, bs):
            idx = perm[i : i + bs]
            key, sk = jax.random.split(key)
            subgraph, opt_state, _ = train_step(
                subgraph, opt_state, x_tr[idx], y_tr[idx], sk
            )

    key, ek = jax.random.split(key)
    test_mse = float(eval_mse(subgraph, x_te, y_te, ek))
    return test_mse


# ---------------------------------------------------------------------------
# Score a subgraph (Schur complement, training-free)
# ---------------------------------------------------------------------------


def score_subgraph(
    graph: ppc.Graph,
    keep: list[int],
    all_J: dict,
    all_r: dict,
    boundary_idx: list[int],
    B_dims: jnp.ndarray,
    I_dims: jnp.ndarray,
    eps: float,
) -> float:
    result = ppc.score_edge_set(all_J, all_r, boundary_idx, keep, B_dims, I_dims, eps)
    return result["score"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    # 1. Generate task
    key, gk = jax.random.split(key)
    task = generate_task(cfg, gk)
    graph = task["graph"]
    cats = task["categories"]
    x_tr, y_tr = task["x_tr"], task["y_tr"]
    x_te, y_te = task["x_te"], task["y_te"]

    n_edges = len(graph.transforms)
    n_ess = sum(1 for c in cats.values() if c == "essential")
    n_skip = sum(1 for c in cats.values() if c == "skip")
    n_noise = sum(1 for c in cats.values() if c == "noise")
    print(f"Graph: {n_edges} edges ({n_ess} essential, {n_skip} skip, {n_noise} noise)")

    # 2. Precompute Jacobians (single forward pass)
    probe_clamps = {"x": x_tr[: cfg.batch_size], "y": y_tr[: cfg.batch_size]}
    state = ppc.init(graph, probe_clamps, key=key)
    all_J, all_r = ppc.precompute_edge_data(graph, state)
    edges = ppc.classify_edges(graph, state)
    boundary_idx = edges["boundary"]
    internal_idx = edges["internal"]
    B_dims, I_dims = ppc.partition_dims(graph, state, boundary_idx)
    print(
        f"Edges: {len(boundary_idx)} boundary, {len(internal_idx)} internal "
        f"(D_B={len(B_dims)}, D_I={len(I_dims)})"
    )

    # 3. Run backward elimination to get the trajectory
    key, rk = jax.random.split(key)
    reduced, diag = ppc.reduce(graph, probe_clamps, cfg.search, rk)

    # 4. Build subgraph collection:
    #    - Full graph
    #    - Each step of backward elimination
    #    - Random subsets of various sizes
    boundary_set = set(boundary_idx)
    all_indices = list(range(n_edges))
    subgraphs = []  # list of (label, keep_indices)

    # Full graph
    subgraphs.append(("full", all_indices))

    # Elimination trajectory
    active = list(all_indices)
    for step in diag["history"][1:]:  # skip the first entry (full graph)
        removed = step["removed"]
        active = [i for i in active if i != removed]
        subgraphs.append((f"elim-{step['removed_name']}", list(active)))

    # Boundary-only (no internal edges)
    subgraphs.append(("boundary-only", list(boundary_idx)))

    # Random subsets: for each size k ∈ {1, 2, ..., len(internal)}, sample a few
    rng = np.random.RandomState(cfg.seed + 1)
    for _ in range(cfg.n_random_subsets):
        n_keep = rng.randint(0, len(internal_idx) + 1)
        kept_internal = list(rng.choice(internal_idx, size=n_keep, replace=False))
        keep = sorted(list(boundary_idx) + kept_internal)
        subgraphs.append((f"random-{n_keep}", keep))

    print(f"\n{len(subgraphs)} subgraphs to evaluate")

    # 5. Score each (training-free)
    print("\nScoring (training-free)...")
    scores = []
    for label, keep in subgraphs:
        s = score_subgraph(
            graph, keep, all_J, all_r, boundary_idx, B_dims, I_dims, cfg.search.eps
        )
        scores.append(s)

    # 6. Train each and measure test MSE
    print("\nTraining subgraphs...")
    mses = []
    for i, (label, keep) in enumerate(subgraphs):
        key, tk = jax.random.split(key)
        sub = build_subgraph(graph, keep)
        n_int = sum(1 for k in keep if k in set(internal_idx))
        t0 = time.perf_counter()
        mse = train_subgraph(sub, cfg, x_tr, y_tr, x_te, y_te, tk)
        dt = time.perf_counter() - t0
        mses.append(mse)
        cat_str = (
            label
            if label.startswith(("full", "boundary", "elim"))
            else f"rand({n_int}int)"
        )
        print(
            f"  [{i+1}/{len(subgraphs)}] {cat_str}: "
            f"score={scores[i]:.4f}  test_mse={mse:.6f}  ({dt:.1f}s)"
        )

    # 7. Compute correlation
    from scipy.stats import spearmanr, pearsonr

    scores_arr = np.array(scores)
    mses_arr = np.array(mses)
    # Filter out any NaN/Inf
    valid = np.isfinite(scores_arr) & np.isfinite(mses_arr)
    sp_r, sp_p = spearmanr(scores_arr[valid], mses_arr[valid])
    pe_r, pe_p = pearsonr(scores_arr[valid], mses_arr[valid])

    print(f"\n{'='*60}")
    print(f"Spearman r = {sp_r:.3f}  (p = {sp_p:.2e})")
    print(f"Pearson  r = {pe_r:.3f}  (p = {pe_p:.2e})")
    print(f"{'='*60}")

    # 8. Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Score vs test MSE
    ax = axes[0]
    colors = []
    for label, _ in subgraphs:
        if label.startswith("full"):
            colors.append("black")
        elif label.startswith("elim"):
            colors.append("steelblue")
        elif label.startswith("boundary"):
            colors.append("red")
        else:
            colors.append("salmon")

    ax.scatter(
        scores_arr[valid],
        mses_arr[valid],
        c=[colors[i] for i in range(len(colors)) if valid[i]],
        s=40,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("Schur complement score (lower = better structure)")
    ax.set_ylabel("Trained test MSE (lower = better performance)")
    ax.set_title(f"Score predicts performance?  Spearman r={sp_r:.3f}")
    ax.grid(True, alpha=0.3)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="black", label="Full graph"),
            Patch(color="steelblue", label="Elimination steps"),
            Patch(color="red", label="Boundary only"),
            Patch(color="salmon", label="Random subsets"),
        ],
        fontsize=8,
    )

    # Score trajectory during elimination
    ax = axes[1]
    elim_scores = [scores[0]]  # full
    for i, (label, _) in enumerate(subgraphs):
        if label.startswith("elim"):
            elim_scores.append(scores[i])
    ax.plot(range(len(elim_scores)), elim_scores, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("Edges removed")
    ax.set_ylabel("Boundary Schur score")
    ax.set_title("Score during backward elimination")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(cfg.output_dir, "score_vs_mse.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"\nSaved to {cfg.output_dir}/")

    # Save results
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "spearman_r": float(sp_r),  # type: ignore
                "spearman_p": float(sp_p),  # type: ignore
                "pearson_r": float(pe_r),  # type: ignore
                "pearson_p": float(pe_p),  # type: ignore
                "subgraphs": [
                    {"label": label, "score": float(s), "test_mse": float(m)}
                    for (label, _), s, m in zip(subgraphs, scores, mses)
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    cfg = tyro.cli(Cfg)
    main(cfg)
