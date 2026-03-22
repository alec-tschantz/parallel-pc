"""Leverage-score structure search experiment.

Tests the claim: leverage scores from the precision matrix predict
which edges matter for post-training accuracy.

Compares:
- Top-k by leverage (informed selection)
- Random-k (uninformed baseline)
- Bottom-k by leverage (adversarial)
- Iterative reduction with energy-gap stopping
"""

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
# Transforms
# ---------------------------------------------------------------------------


class MLPRelu(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_dim, hid, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, hid, key=k1)
        self.linear2 = eqx.nn.Linear(hid, out_dim, key=k2)

    def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))


class MLPGelu(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_dim, hid, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, hid, key=k1)
        self.linear2 = eqx.nn.Linear(hid, out_dim, key=k2)

    def __call__(self, x):
        return self.linear2(jax.nn.gelu(self.linear1(x)))


class Conv2dBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    proj: eqx.nn.Linear
    ph: int = eqx.field(static=True)
    pw: int = eqx.field(static=True)

    def __init__(self, ph, pw, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.ph, self.pw = ph, pw
        self.conv = eqx.nn.Conv2d(1, 8, 3, key=k1)
        self.proj = eqx.nn.Linear(8 * (ph - 2) * (pw - 2), out_dim, key=k2)

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
    n_globals: int = 1
    mlp_hidden: int = 32
    search: ppc.SearchConfig = field(default_factory=ppc.SearchConfig)
    ks: str = "5,8,12"  # edge counts to test
    n_random_trials: int = 5
    train_epochs: int = 20
    train_lr: float = 1e-4
    train_batch_size: int = 64
    eval_every: int = 5
    output_dir: str = "outputs/search"


# ---------------------------------------------------------------------------
# Data + graph construction
# ---------------------------------------------------------------------------


def load_data(cfg):
    (x_tr, y_tr), (x_te, y_te) = fmnist_data.load_data()
    x_val, y_val = x_tr[-cfg.val_size:], y_tr[-cfg.val_size:]
    x_tr, y_tr = x_tr[:-cfg.val_size], y_tr[:-cfg.val_size]
    grid = int(cfg.n_patches**0.5)

    def extract(imgs):
        ps = []
        for r in range(grid):
            for c in range(grid):
                p = imgs[:, r*cfg.patch_h:(r+1)*cfg.patch_h, c*cfg.patch_w:(c+1)*cfg.patch_w]
                ps.append(jnp.array(p.reshape(len(imgs), -1).astype("float32") / 255.0))
        return ps[:cfg.n_patches]
    return (extract(x_tr), jax.nn.one_hot(y_tr, 10),
            extract(x_val), jax.nn.one_hot(y_val, 10),
            extract(x_te), jax.nn.one_hot(y_te, 10))


def make_variables(cfg):
    pd = cfg.patch_h * cfg.patch_w
    vs = [ppc.Variable("output", (10,))]
    for i in range(cfg.n_patches):
        vs.append(ppc.Variable(f"p_{i}", (pd,)))
        vs.append(ppc.Variable(f"h_local_{i}", (cfg.local_dim,)))
    for i in range(cfg.n_regions):
        vs.append(ppc.Variable(f"h_region_{i}", (cfg.region_dim,)))
    for i in range(cfg.n_globals):
        vs.append(ppc.Variable(f"h_global_{i}", (cfg.global_dim,)))
    return vs


def make_candidates(cfg):
    pd = cfg.patch_h * cfg.patch_w
    cs = []
    L = lambda i, o, s, t: lambda k: ppc.Transform("_", eqx.nn.Linear(i, o, key=k), src=s, tgt=t)
    MR = lambda i, o, s, t: lambda k: ppc.Transform("_", MLPRelu(i, cfg.mlp_hidden, o, key=k), src=s, tgt=t)
    MG = lambda i, o, s, t: lambda k: ppc.Transform("_", MLPGelu(i, cfg.mlp_hidden, o, key=k), src=s, tgt=t)
    CV = lambda o, s, t: lambda k: ppc.Transform("_", Conv2dBlock(cfg.patch_h, cfg.patch_w, o, key=k), src=s, tgt=t)
    mse = lambda tgt: lambda tid: ppc.Energy(ppc.mse_energy, args=[tid, tgt])
    ce = lambda tid: ppc.Energy(ppc.cross_entropy_energy, args=[tid, "output"])

    for i in range(cfg.n_patches):
        s, t = f"p_{i}", f"h_local_{i}"
        cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(pd, cfg.local_dim, s, t), mse(t)))
        cs.append(ppc.Candidate(f"{s}->{t}[MLPRelu]", MR(pd, cfg.local_dim, s, t), mse(t)))
        if cfg.patch_h >= 5:
            cs.append(ppc.Candidate(f"{s}->{t}[Conv2d]", CV(cfg.local_dim, s, t), mse(t)))

    ppr = int(cfg.n_patches**0.5)
    rpr = int(cfg.n_regions**0.5) if cfg.n_regions > 1 else 1
    for pi in range(cfg.n_patches):
        ri = min((pi // ppr) // max(ppr // rpr, 1), rpr - 1) * rpr + min((pi % ppr) // max(ppr // rpr, 1), rpr - 1)
        ri = min(ri, cfg.n_regions - 1)
        s, t = f"h_local_{pi}", f"h_region_{ri}"
        cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(cfg.local_dim, cfg.region_dim, s, t), mse(t)))

    for i in range(cfg.n_regions):
        for j in range(cfg.n_globals):
            s, t = f"h_region_{i}", f"h_global_{j}"
            cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(cfg.region_dim, cfg.global_dim, s, t), mse(t)))
            cs.append(ppc.Candidate(f"{s}->{t}[MLPGelu]", MG(cfg.region_dim, cfg.global_dim, s, t), mse(t)))

    for i in range(cfg.n_globals):
        for j in range(cfg.n_globals):
            if i != j:
                s, t = f"h_global_{i}", f"h_global_{j}"
                cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(cfg.global_dim, cfg.global_dim, s, t), mse(t)))

    for i in range(cfg.n_regions):
        s = f"h_region_{i}"
        cs.append(ppc.Candidate(f"{s}->output[Linear]", L(cfg.region_dim, 10, s, "output"), ce))
        cs.append(ppc.Candidate(f"{s}->output[MLPRelu]", MR(cfg.region_dim, 10, s, "output"), ce))
    for j in range(cfg.n_globals):
        s = f"h_global_{j}"
        cs.append(ppc.Candidate(f"{s}->output[Linear]", L(cfg.global_dim, 10, s, "output"), ce))
        cs.append(ppc.Candidate(f"{s}->output[MLPGelu]", MG(cfg.global_dim, 10, s, "output"), ce))

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


def train_and_eval(graph, cfg, patches_tr, y_tr, patches_te, y_te, key):
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
            c = make_clamps(patches_tr, y_tr, perm[i:i+bs], cfg.n_patches)
            s = ppc.init(graph, c, key=sk)
            s = ppc.infer(graph, s, optimizer=iopt, iters=cfg.search.infer_iters)
            loss = ppc.energy(graph, s)
            grads = ppc.param_grad(graph, s)
            updates, ostate = topt.update(eqx.filter(grads, eqx.is_array), ostate, eqx.filter(graph, eqx.is_array))
            graph = eqx.apply_updates(graph, updates)
            tloss += float(loss); n += 1
        dt = time.perf_counter() - t0
        if ep % cfg.eval_every == 0 or ep == cfg.train_epochs:
            corr, tot = 0, 0
            for i in range(0, y_te.shape[0] - bs + 1, bs):
                key, ek2 = jax.random.split(key)
                c = make_clamps_nolabel(patches_te, jnp.arange(i, i+bs), cfg.n_patches)
                s = ppc.init(graph, c, key=ek2)
                s = ppc.infer(graph, s, optimizer=iopt, iters=cfg.search.infer_iters * 2)
                p = ppc.variable(graph, s, "output")
                corr += int(jnp.sum(jnp.argmax(p, -1) == jnp.argmax(y_te[i:i+bs], -1)))
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

    # Compute leverage scores
    key, bk = jax.random.split(key)
    batch_idx = jax.random.permutation(bk, y_val.shape[0])[:min(256, y_val.shape[0])]
    clamps_true = make_clamps(p_val, y_val, batch_idx, cfg.n_patches)

    print("\nComputing leverage scores...")
    key, lk = jax.random.split(key)
    levs, M_inv = ppc.compute_leverages(supergraph, clamps_true, cfg.search, lk)

    # Print leverage ranking
    ranked = sorted(range(len(levs)), key=lambda i: levs[i], reverse=True)
    print(f"\nLeverage ranking (all {len(levs)} edges):")
    for r, i in enumerate(ranked):
        print(f"  {r+1:2d}. {supergraph.transforms[i].id:50s} leverage={levs[i]:.4f}")

    # --- Run iterative reduction with energy-gap stopping ---
    print(f"\n{'='*60}")
    print("ITERATIVE REDUCTION (leverage + energy-gap stopping)")
    print(f"{'='*60}")
    key, wk = jax.random.split(key)
    wrong_perm = jax.random.permutation(wk, len(batch_idx))
    clamps_wrong = {**clamps_true, "output": clamps_true["output"][wrong_perm]}

    key, rk = jax.random.split(key)
    reduced, diag = ppc.reduce(supergraph, clamps_true, clamps_wrong, cfg.search, rk)
    n_reduced = len(reduced.transforms)
    print(f"Reduced to {n_reduced} edges")

    # --- Leverage-vs-random ranking experiment ---
    ks = [int(k) for k in cfg.ks.split(",")]
    results = {}

    for k in ks:
        if k > len(supergraph.transforms):
            continue
        print(f"\n{'='*60}")
        print(f"k = {k} edges")
        print(f"{'='*60}")

        # Top-k by leverage
        top_k = ppc.select_top_k(supergraph, levs, k)
        top_graph = ppc.search._build_subgraph(supergraph, top_k)
        top_names = [supergraph.transforms[i].id for i in top_k]
        print(f"  Top-{k}: {[n.split('_', 2)[-1] for n in top_names]}")
        key, tk = jax.random.split(key)
        print(f"  Training top-{k}...")
        top_accs = train_and_eval(top_graph, cfg, p_tr, y_tr, p_te, y_te, tk)

        # Bottom-k by leverage
        bottom_k = sorted(ranked[-k:])
        bot_graph = ppc.search._build_subgraph(supergraph, bottom_k)
        bot_names = [supergraph.transforms[i].id for i in bottom_k]
        print(f"  Bottom-{k}: {[n.split('_', 2)[-1] for n in bot_names]}")
        key, bk2 = jax.random.split(key)
        print(f"  Training bottom-{k}...")
        bot_accs = train_and_eval(bot_graph, cfg, p_tr, y_tr, p_te, y_te, bk2)

        # Random-k
        rand_accs_list = []
        for trial in range(cfg.n_random_trials):
            key, rk2 = jax.random.split(key)
            rand_graph = ppc.random_reduce(supergraph, k, rk2)
            key, tk2 = jax.random.split(key)
            print(f"  Training random-{k} (trial {trial+1})...")
            ra = train_and_eval(rand_graph, cfg, p_tr, y_tr, p_te, y_te, tk2)
            rand_accs_list.append(ra)

        results[k] = {
            "top": {"edges": top_names, "accs": top_accs},
            "bottom": {"edges": bot_names, "accs": bot_accs},
            "random": [{"accs": ra} for ra in rand_accs_list],
        }

    # Also train the iteratively reduced graph
    if n_reduced > 0:
        print(f"\n  Training iteratively-reduced ({n_reduced} edges)...")
        key, tk3 = jax.random.split(key)
        red_accs = train_and_eval(reduced, cfg, p_tr, y_tr, p_te, y_te, tk3)
    else:
        red_accs = []

    # --- Plot ---
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. Accuracy vs k
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in sorted(results.keys()):
        r = results[k]
        top_best = max(r["top"]["accs"]) if r["top"]["accs"] else 0
        bot_best = max(r["bottom"]["accs"]) if r["bottom"]["accs"] else 0
        rand_bests = [max(ra["accs"]) if ra["accs"] else 0 for ra in r["random"]]
        ax.scatter(k, top_best, color="blue", s=100, zorder=5, marker="^")
        ax.scatter(k, bot_best, color="red", s=100, zorder=5, marker="v")
        if rand_bests:
            ax.scatter([k]*len(rand_bests), rand_bests, color="gray", s=30, alpha=0.5)
            ax.scatter(k, np.mean(rand_bests), color="orange", s=80, zorder=4, marker="s")
    ax.plot([], [], "b^", label="Top-k (leverage)")
    ax.plot([], [], "rv", label="Bottom-k (leverage)")
    ax.plot([], [], "s", color="orange", label="Random-k (mean)")
    if red_accs:
        ax.axhline(max(red_accs), color="green", linestyle="--", label=f"Iterative reduction ({n_reduced} edges)")
    ax.set_xlabel("Number of edges (k)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Leverage Ranking vs Random Selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "accuracy_vs_k.png"), dpi=150)
    plt.close()

    # 2. Leverage distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(levs)), [levs[i] for i in ranked], color="steelblue")
    ax.set_xlabel("Edge rank")
    ax.set_ylabel("Leverage score")
    ax.set_title("Leverage Score Distribution (sorted)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "leverage_distribution.png"), dpi=150)
    plt.close()

    # Save
    output = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "leverage_ranking": [(supergraph.transforms[i].id, levs[i]) for i in ranked],
        "reduction": diag,
        "reduced_accs": red_accs,
        "k_results": {str(k): {
            "top_acc": max(r["top"]["accs"]) if r["top"]["accs"] else 0,
            "bottom_acc": max(r["bottom"]["accs"]) if r["bottom"]["accs"] else 0,
            "random_accs": [max(ra["accs"]) if ra["accs"] else 0 for ra in r["random"]],
        } for k, r in results.items()},
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for k in sorted(results.keys()):
        r = results[k]
        top = max(r["top"]["accs"]) if r["top"]["accs"] else 0
        bot = max(r["bottom"]["accs"]) if r["bottom"]["accs"] else 0
        rand = [max(ra["accs"]) if ra["accs"] else 0 for ra in r["random"]]
        print(f"  k={k:2d}: top={top:.4f}  bottom={bot:.4f}  "
              f"random={np.mean(rand):.4f}±{np.std(rand):.4f}")
    if red_accs:
        print(f"  Iterative reduction ({n_reduced} edges): {max(red_accs):.4f}")
    print(f"\nOutputs: {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(Cfg)
    main(cfg)
