import json
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("KERAS_BACKEND", "jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from keras.datasets import fashion_mnist as fmnist_data

import ppc


# ---------------------------------------------------------------------------
# Transform modules
# ---------------------------------------------------------------------------


class MLPRelu(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, i, h, o, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(i, h, key=k1)
        self.linear2 = eqx.nn.Linear(h, o, key=k2)

    def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))


class Conv2dBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    proj: eqx.nn.Linear
    ph: int = eqx.field(static=True)
    pw: int = eqx.field(static=True)

    def __init__(self, ph, pw, o, *, key):
        k1, k2 = jax.random.split(key)
        self.ph, self.pw = ph, pw
        self.conv = eqx.nn.Conv2d(1, 8, 3, key=k1)
        self.proj = eqx.nn.Linear(8 * (ph - 2) * (pw - 2), o, key=k2)

    def __call__(self, x):
        return self.proj(jax.nn.relu(self.conv(x.reshape(1, self.ph, self.pw))).ravel())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Cfg:
    seed: int = 0
    patch_h: int = 14
    patch_w: int = 14
    n_patches: int = 4
    val_size: int = 5000
    local_dim: int = 32
    region_dim: int = 48
    global_dim: int = 64
    n_regions: int = 2
    n_globals: int = 2
    mlp_hidden: int = 32
    search: ppc.SearchConfig = field(default_factory=ppc.SearchConfig)
    n_random_trials: int = 3
    train_epochs: int = 15
    train_lr: float = 1e-4
    train_batch_size: int = 64
    eval_every: int = 5
    output_dir: str = "outputs/search"


# ---------------------------------------------------------------------------
# Data + graph
# ---------------------------------------------------------------------------


def load_data(cfg):
    (x_tr, y_tr), (x_te, y_te) = fmnist_data.load_data()
    x_val, y_val = x_tr[-cfg.val_size :], y_tr[-cfg.val_size :]
    x_tr, y_tr = x_tr[: -cfg.val_size], y_tr[: -cfg.val_size]
    grid = int(cfg.n_patches**0.5)

    def extract(imgs):
        return [
            jnp.array(
                imgs[
                    :,
                    r * cfg.patch_h : (r + 1) * cfg.patch_h,
                    c * cfg.patch_w : (c + 1) * cfg.patch_w,
                ]
                .reshape(len(imgs), -1)
                .astype("float32")
                / 255.0
            )
            for r in range(grid)
            for c in range(grid)
        ][: cfg.n_patches]

    return (
        extract(x_tr),
        jax.nn.one_hot(y_tr, 10),
        extract(x_val),
        jax.nn.one_hot(y_val, 10),
        extract(x_te),
        jax.nn.one_hot(y_te, 10),
    )


def make_variables(cfg):
    pd = cfg.patch_h * cfg.patch_w
    vs = [ppc.Variable("output", (10,))]
    for i in range(cfg.n_patches):
        vs += [
            ppc.Variable(f"p_{i}", (pd,)),
            ppc.Variable(f"h_local_{i}", (cfg.local_dim,)),
        ]
    for i in range(cfg.n_regions):
        vs.append(ppc.Variable(f"h_region_{i}", (cfg.region_dim,)))
    for i in range(cfg.n_globals):
        vs.append(ppc.Variable(f"h_global_{i}", (cfg.global_dim,)))
    return vs


def make_candidates(cfg):
    pd = cfg.patch_h * cfg.patch_w
    cs = []
    L = lambda i, o, s, t: lambda k: ppc.Transform(
        "_", eqx.nn.Linear(i, o, key=k), src=s, tgt=t
    )
    MR = lambda i, h, o, s, t: lambda k: ppc.Transform(
        "_", MLPRelu(i, h, o, key=k), src=s, tgt=t
    )
    CV = lambda o, s, t: lambda k: ppc.Transform(
        "_", Conv2dBlock(cfg.patch_h, cfg.patch_w, o, key=k), src=s, tgt=t
    )
    mse = lambda tgt: lambda tid: ppc.Energy(ppc.mse_energy, args=[tid, tgt])
    ce = lambda tid: ppc.Energy(ppc.cross_entropy_energy, args=[tid, "output"])

    # patch → local (BOUNDARY: p_i is clamped)
    for i in range(cfg.n_patches):
        s, t = f"p_{i}", f"h_local_{i}"
        cs.append(
            ppc.Candidate(f"{s}->{t}[Linear]", L(pd, cfg.local_dim, s, t), mse(t))
        )
        cs.append(
            ppc.Candidate(
                f"{s}->{t}[MLPRelu]",
                MR(pd, cfg.mlp_hidden, cfg.local_dim, s, t),
                mse(t),
            )
        )
        if cfg.patch_h >= 5:
            cs.append(
                ppc.Candidate(f"{s}->{t}[Conv2d]", CV(cfg.local_dim, s, t), mse(t))
            )

    # local → region (INTERNAL: both free) — multiple transform types
    ppr = int(cfg.n_patches**0.5)
    rpr = int(cfg.n_regions**0.5) if cfg.n_regions > 1 else 1
    for pi in range(cfg.n_patches):
        ri = min((pi // ppr) // max(ppr // rpr, 1), rpr - 1) * rpr + min(
            (pi % ppr) // max(ppr // rpr, 1), rpr - 1
        )
        ri = min(ri, cfg.n_regions - 1)
        s, t = f"h_local_{pi}", f"h_region_{ri}"
        cs.append(
            ppc.Candidate(
                f"{s}->{t}[Linear]", L(cfg.local_dim, cfg.region_dim, s, t), mse(t)
            )
        )
        cs.append(
            ppc.Candidate(
                f"{s}->{t}[MLPRelu]",
                MR(cfg.local_dim, cfg.mlp_hidden, cfg.region_dim, s, t),
                mse(t),
            )
        )

    # local → region cross-connections (connect to OTHER region too)
    for pi in range(cfg.n_patches):
        ri = min((pi // ppr) // max(ppr // rpr, 1), rpr - 1) * rpr + min(
            (pi % ppr) // max(ppr // rpr, 1), rpr - 1
        )
        ri = min(ri, cfg.n_regions - 1)
        for rj in range(cfg.n_regions):
            if rj != ri:
                s, t = f"h_local_{pi}", f"h_region_{rj}"
                cs.append(
                    ppc.Candidate(
                        f"{s}->{t}[Linear]",
                        L(cfg.local_dim, cfg.region_dim, s, t),
                        mse(t),
                    )
                )

    # region → global (INTERNAL) — multiple types
    for i in range(cfg.n_regions):
        for j in range(cfg.n_globals):
            s, t = f"h_region_{i}", f"h_global_{j}"
            cs.append(
                ppc.Candidate(
                    f"{s}->{t}[Linear]", L(cfg.region_dim, cfg.global_dim, s, t), mse(t)
                )
            )
            cs.append(
                ppc.Candidate(
                    f"{s}->{t}[MLPRelu]",
                    MR(cfg.region_dim, cfg.mlp_hidden, cfg.global_dim, s, t),
                    mse(t),
                )
            )

    # global ↔ global (INTERNAL)
    for i in range(cfg.n_globals):
        for j in range(cfg.n_globals):
            if i != j:
                s, t = f"h_global_{i}", f"h_global_{j}"
                cs.append(
                    ppc.Candidate(
                        f"{s}->{t}[Linear]",
                        L(cfg.global_dim, cfg.global_dim, s, t),
                        mse(t),
                    )
                )

    # → output (BOUNDARY: output is clamped)
    for i in range(cfg.n_regions):
        s = f"h_region_{i}"
        cs.append(
            ppc.Candidate(
                f"{s}->output[Linear]", L(cfg.region_dim, 10, s, "output"), ce
            )
        )
    for j in range(cfg.n_globals):
        s = f"h_global_{j}"
        cs.append(
            ppc.Candidate(
                f"{s}->output[Linear]", L(cfg.global_dim, 10, s, "output"), ce
            )
        )

    return cs


def make_clamps(patches, y, idx, np_):
    c = {"output": y[idx]}
    for i in range(np_):
        c[f"p_{i}"] = patches[i][idx]
    return c


def make_clamps_nolabel(patches, idx, np_):
    return {f"p_{i}": patches[i][idx] for i in range(np_)}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_and_eval(graph, cfg, p_tr, y_tr, p_te, y_te, key):
    iopt = optax.adam(cfg.search.infer_lr)
    topt = optax.adam(cfg.train_lr)
    ostate = topt.init(eqx.filter(graph, eqx.is_array))
    bs = cfg.train_batch_size
    accs = []
    for ep in range(1, cfg.train_epochs + 1):
        t0 = time.perf_counter()
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, y_tr.shape[0])
        tloss, n = 0.0, 0
        for i in range(0, y_tr.shape[0] - bs + 1, bs):
            key, sk = jax.random.split(key)
            c = make_clamps(p_tr, y_tr, perm[i : i + bs], cfg.n_patches)
            s = ppc.init(graph, c, key=sk)
            s = ppc.infer(graph, s, optimizer=iopt, iters=cfg.search.infer_iters)
            grads = ppc.param_grad(graph, s)
            tloss += float(ppc.energy(graph, s))
            n += 1
            updates, ostate = topt.update(
                eqx.filter(grads, eqx.is_array), ostate, eqx.filter(graph, eqx.is_array)
            )
            graph = eqx.apply_updates(graph, updates)
        dt = time.perf_counter() - t0
        if ep % cfg.eval_every == 0 or ep == cfg.train_epochs:
            corr, tot = 0, 0
            for i in range(0, y_te.shape[0] - bs + 1, bs):
                key, ek2 = jax.random.split(key)
                c = make_clamps_nolabel(p_te, jnp.arange(i, i + bs), cfg.n_patches)
                s = ppc.init(graph, c, key=ek2)
                s = ppc.infer(
                    graph, s, optimizer=iopt, iters=cfg.search.infer_iters * 2
                )
                p = ppc.variable(graph, s, "output")
                corr += int(
                    jnp.sum(jnp.argmax(p, -1) == jnp.argmax(y_te[i : i + bs], -1))
                )
                tot += bs
            acc = corr / tot if tot > 0 else 0.0
            accs.append(acc)
            print(f"    Ep {ep:3d}  loss={tloss/n:.2f}  acc={acc:.4f}  ({dt:.1f}s)")
        else:
            print(f"    Ep {ep:3d}  loss={tloss/n:.2f}  ({dt:.1f}s)")
    return accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    print("Loading Fashion-MNIST patches...")
    p_tr, y_tr, p_val, y_val, p_te, y_te = load_data(cfg)

    variables = make_variables(cfg)
    candidates = make_candidates(cfg)
    print(f"Variables: {len(variables)}  Candidates: {len(candidates)}")

    # Build supergraph
    key, gk = jax.random.split(key)
    base = ppc.Graph(variables=variables, transforms=[], energies=[])
    supergraph = ppc.build_supergraph(base, candidates, gk)
    print(f"Supergraph: {len(supergraph.transforms)} edges")

    # Search clamps from validation
    key, bk = jax.random.split(key)
    batch_idx = jax.random.permutation(bk, y_val.shape[0])[: min(256, y_val.shape[0])]
    clamps = make_clamps(p_val, y_val, batch_idx, cfg.n_patches)

    # ===== PHASE 1: VALIDATE THEORY =====
    print(f"\n{'='*60}")
    print("PHASE 1: Validate theory (monotonicity of φ_T^B)")
    print(f"{'='*60}")

    key, rk = jax.random.split(key)
    reduced, diag = ppc.reduce(supergraph, clamps, cfg.search, rk)

    # Check monotonicity
    phis = [h["phi_T_B"] for h in diag["history"]]
    monotonic = all(phis[i] <= phis[i + 1] + 1e-6 for i in range(len(phis) - 1))
    print(f"\n  φ_T^B values: {[f'{p:.4f}' for p in phis]}")
    print(
        f"  Monotonic: {'YES ✓' if monotonic else 'NO ✗ — frozen-RHS fix may be incorrect'}"
    )
    print(f"  Boundary edges (kept): {diag['n_boundary']}")
    print(f"  Internal edges: {diag['n_internal_start']} → {diag['n_internal_final']}")
    print(f"  Final edges: {[t.id for t in reduced.transforms]}")

    # Plot 1: φ_T^B during pruning
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    n_edges = [h["n_total"] for h in diag["history"]]
    axes[0].plot(n_edges, phis, "b-o", linewidth=2)
    axes[0].set_xlabel("Edges remaining")
    axes[0].set_ylabel("φ_T^B")
    axes[0].set_title("Boundary φ_T during pruning" + (" ✓" if monotonic else " ✗"))
    axes[0].grid(True, alpha=0.3)

    # Plot 2: coverage gap + conditioning penalty
    cg = [h["coverage_gap"] for h in diag["history"]]
    cp = [h["conditioning_penalty"] for h in diag["history"]]
    axes[1].plot(n_edges, cg, "r-s", label="Coverage gap", linewidth=2)
    axes[1].plot(n_edges, cp, "g-^", label="Cond. penalty", linewidth=2)
    axes[1].set_xlabel("Edges remaining")
    axes[1].set_title("Score decomposition")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: leverage distribution
    if diag["history"] and "leverages" in diag["history"][-1]:
        all_levs = {}
        for h in diag["history"][1:]:
            if "leverages" in h:
                all_levs.update(h["leverages"])
        if all_levs:
            vals = list(all_levs.values())
            axes[2].hist(vals, bins=20, color="steelblue", alpha=0.7)
            axes[2].set_xlabel("Leverage score")
            axes[2].set_title("Leverage of pruned edges")
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "phase1_theory.png"), dpi=150)
    plt.close()

    if not monotonic:
        violations = [
            (i, phis[i + 1] - phis[i])
            for i in range(len(phis) - 1)
            if phis[i + 1] < phis[i] - 1e-6
        ]
        print(
            f"\n  ⚠ MONOTONICITY VIOLATED at {len(violations)} point(s): {violations}"
        )
        print(
            "  (Small violations at near-empty graphs are expected from linearisation looseness)"
        )
        # Don't stop — small violations are a known limitation

    # ===== PHASE 2: VALIDATE SCORE =====
    print(f"\n{'='*60}")
    print("PHASE 2: Validate score (leverage-pruned vs random-pruned)")
    print(f"{'='*60}")

    n_internal_final = diag["n_internal_final"]
    print(f"  Leverage-pruned: {n_internal_final} internal edges")

    # Train leverage-pruned
    print(f"\n  Training leverage-pruned...")
    key, tk = jax.random.split(key)
    lev_accs = train_and_eval(reduced, cfg, p_tr, y_tr, p_te, y_te, tk)

    # Train random-pruned (same number of internal edges)
    # Need state for classify_edges
    key, sk = jax.random.split(key)
    state_for_class = ppc.init(supergraph, clamps, key=sk)
    state_for_class = ppc.infer(
        supergraph,
        state_for_class,
        optimizer=optax.adam(cfg.search.infer_lr),
        iters=cfg.search.infer_iters,
    )

    rand_accs_list = []
    for trial in range(cfg.n_random_trials):
        print(f"\n  Training random-pruned (trial {trial+1})...")
        key, rk2 = jax.random.split(key)
        rand_graph = ppc.random_reduce(
            supergraph, state_for_class, n_internal_final, rk2
        )
        key, tk2 = jax.random.split(key)
        ra = train_and_eval(rand_graph, cfg, p_tr, y_tr, p_te, y_te, tk2)
        rand_accs_list.append(ra)

    lev_best = max(lev_accs) if lev_accs else 0
    rand_bests = [max(ra) if ra else 0 for ra in rand_accs_list]

    print(f"\n  Leverage-pruned: {lev_best:.4f}")
    print(f"  Random-pruned:   {np.mean(rand_bests):.4f} ± {np.std(rand_bests):.4f}")
    print(f"  Gap: {lev_best - np.mean(rand_bests):+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [rand_bests],
        positions=[2],
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor="salmon", alpha=0.7),
    )
    ax.scatter(
        [1],
        [lev_best],
        color="steelblue",
        s=200,
        zorder=5,
        marker="D",
        label="Leverage-pruned",
    )
    for i, rb in enumerate(rand_bests):
        ax.scatter(2 + np.random.uniform(-0.1, 0.1), rb, color="gray", s=30, alpha=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Leverage", "Random"])
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy ({n_internal_final} internal edges)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "phase2_score.png"), dpi=150)
    plt.close()

    # ===== SAVE =====
    output = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "phase1": {
            "monotonic": monotonic,
            "phi_B_values": phis,
            "n_edges": n_edges,
            "diag": {k: v for k, v in diag.items() if k != "history"},
        },
        "phase2": {"leverage_acc": lev_best, "random_accs": rand_bests},
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(
            output,
            f,
            indent=2,
            default=lambda o: float(o) if hasattr(o, "__float__") else str(o),
        )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 1: {'PASS' if monotonic else 'FAIL'}")
    print(
        f"  Phase 2: leverage={lev_best:.4f}, random={np.mean(rand_bests):.4f}±{np.std(rand_bests):.4f}"
    )
    print(f"  Outputs: {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(Cfg)
    main(cfg)
