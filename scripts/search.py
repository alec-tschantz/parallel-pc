"""Structure search experiment: leverage-score reduction vs random pruning.

Builds a supergraph with all candidate edges on Fashion-MNIST patches,
then prunes via leverage scores (Algorithm 1) and compares to random pruning.
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
# Transform modules
# ---------------------------------------------------------------------------


class MLPRelu(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=k2)

    def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))


class MLPGelu(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=k2)

    def __call__(self, x):
        return self.linear2(jax.nn.gelu(self.linear1(x)))


class Conv2dBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    proj: eqx.nn.Linear
    patch_h: int = eqx.field(static=True)
    patch_w: int = eqx.field(static=True)

    def __init__(self, patch_h, patch_w, out_dim, n_filters=8, *, key):
        k1, k2 = jax.random.split(key)
        self.patch_h, self.patch_w = patch_h, patch_w
        self.conv = eqx.nn.Conv2d(1, n_filters, 3, key=k1)
        self.proj = eqx.nn.Linear(n_filters * (patch_h - 2) * (patch_w - 2), out_dim, key=k2)

    def __call__(self, x):
        return self.proj(jax.nn.relu(self.conv(x.reshape(1, self.patch_h, self.patch_w))).ravel())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
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
    n_reduce_trials: int = 2
    n_random_trials: int = 5
    train_epochs: int = 30
    train_lr: float = 1e-4
    train_batch_size: int = 64
    eval_every: int = 10
    output_dir: str = "outputs/search"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(cfg):
    (x_tr, y_tr), (x_te, y_te) = fmnist_data.load_data()
    x_val, y_val = x_tr[-cfg.val_size:], y_tr[-cfg.val_size:]
    x_tr, y_tr = x_tr[:-cfg.val_size], y_tr[:-cfg.val_size]
    grid = int(cfg.n_patches**0.5)
    def extract(imgs):
        patches = []
        for r in range(grid):
            for c in range(grid):
                p = imgs[:, r*cfg.patch_h:(r+1)*cfg.patch_h, c*cfg.patch_w:(c+1)*cfg.patch_w]
                patches.append(jnp.array(p.reshape(len(imgs), -1).astype("float32") / 255.0))
        return patches[:cfg.n_patches]
    return (extract(x_tr), jax.nn.one_hot(y_tr, 10),
            extract(x_val), jax.nn.one_hot(y_val, 10),
            extract(x_te), jax.nn.one_hot(y_te, 10))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def make_variables(cfg):
    pd = cfg.patch_h * cfg.patch_w
    vs = [ppc.Variable("output", (10,))]
    for i in range(cfg.n_patches):
        vs.append(ppc.Variable(f"p_{i}", (pd,)))
    for i in range(cfg.n_patches):
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

    # patch → local
    for i in range(cfg.n_patches):
        s, t = f"p_{i}", f"h_local_{i}"
        cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(pd, cfg.local_dim, s, t), mse(t)))
        cs.append(ppc.Candidate(f"{s}->{t}[MLPRelu]", MR(pd, cfg.local_dim, s, t), mse(t)))
        if cfg.patch_h >= 5 and cfg.patch_w >= 5:
            cs.append(ppc.Candidate(f"{s}->{t}[Conv2d]", CV(cfg.local_dim, s, t), mse(t)))

    # local → region
    ppr = int(cfg.n_patches**0.5)
    rpr = int(cfg.n_regions**0.5) if cfg.n_regions > 1 else 1
    for pi in range(cfg.n_patches):
        ri = min((pi // ppr) // max(ppr // rpr, 1), rpr - 1) * rpr + min((pi % ppr) // max(ppr // rpr, 1), rpr - 1)
        ri = min(ri, cfg.n_regions - 1)
        s, t = f"h_local_{pi}", f"h_region_{ri}"
        cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(cfg.local_dim, cfg.region_dim, s, t), mse(t)))

    # region → global
    for i in range(cfg.n_regions):
        for j in range(cfg.n_globals):
            s, t = f"h_region_{i}", f"h_global_{j}"
            cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(cfg.region_dim, cfg.global_dim, s, t), mse(t)))
            cs.append(ppc.Candidate(f"{s}->{t}[MLPGelu]", MG(cfg.region_dim, cfg.global_dim, s, t), mse(t)))

    # global ↔ global
    for i in range(cfg.n_globals):
        for j in range(cfg.n_globals):
            if i != j:
                s, t = f"h_global_{i}", f"h_global_{j}"
                cs.append(ppc.Candidate(f"{s}->{t}[Linear]", L(cfg.global_dim, cfg.global_dim, s, t), mse(t)))

    # → output
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
# Training
# ---------------------------------------------------------------------------


def train_and_evaluate(graph, cfg, patches_tr, y_tr, patches_te, y_te, key):
    infer_opt = optax.adam(cfg.search.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))
    bs = cfg.train_batch_size
    accs = []
    for epoch in range(1, cfg.train_epochs + 1):
        t0 = time.perf_counter()
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, y_tr.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, y_tr.shape[0] - bs + 1, bs):
            key, sk = jax.random.split(key)
            c = make_clamps(patches_tr, y_tr, perm[i:i+bs], cfg.n_patches)
            s = ppc.init(graph, c, key=sk)
            s = ppc.infer(graph, s, optimizer=infer_opt, iters=cfg.search.infer_iters)
            loss = ppc.energy(graph, s)
            grads = ppc.param_grad(graph, s)
            updates, opt_state = train_opt.update(
                eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(graph, eqx.is_array))
            graph = eqx.apply_updates(graph, updates)
            total_loss += float(loss)
            n += 1
        dt = time.perf_counter() - t0
        if epoch % cfg.eval_every == 0 or epoch == cfg.train_epochs:
            correct, total = 0, 0
            for i in range(0, y_te.shape[0] - bs + 1, bs):
                key, ek2 = jax.random.split(key)
                c = make_clamps_nolabel(patches_te, jnp.arange(i, i+bs), cfg.n_patches)
                s = ppc.init(graph, c, key=ek2)
                s = ppc.infer(graph, s, optimizer=infer_opt, iters=cfg.search.infer_iters * 2)
                preds = ppc.variable(graph, s, "output")
                correct += int(jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y_te[i:i+bs], -1)))
                total += bs
            acc = correct / total if total > 0 else 0.0
            accs.append(acc)
            print(f"    Epoch {epoch:3d}  loss={total_loss/n:.2f}  acc={acc:.4f}  ({dt:.1f}s)")
        else:
            print(f"    Epoch {epoch:3d}  loss={total_loss/n:.2f}  ({dt:.1f}s)")
    return accs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_leverage_distribution(diagnostics, output_dir):
    """Leverage scores of retained vs pruned edges."""
    if not diagnostics.get("history"):
        return
    initial = diagnostics["history"][0].get("leverage_scores", {})
    final_ids = set(diagnostics.get("final_leverage", {}).keys())
    retained = [v for k, v in initial.items() if k in final_ids]
    pruned = [v for k, v in initial.items() if k not in final_ids]
    if not retained and not pruned:
        return
    _, ax = plt.subplots(figsize=(8, 4))
    if retained:
        ax.hist(retained, bins=20, alpha=0.7, label=f"Retained ({len(retained)})", color="steelblue")
    if pruned:
        ax.hist(pruned, bins=20, alpha=0.7, label=f"Pruned ({len(pruned)})", color="salmon")
    ax.set_xlabel("Leverage score")
    ax.set_ylabel("Count")
    ax.set_title("Leverage Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "leverage_distribution.png"), dpi=150)
    plt.close()


def plot_phi_progression(diagnostics, output_dir):
    """φ_T during pruning rounds."""
    hist = diagnostics.get("history", [])
    if len(hist) < 2:
        return
    n_edges = [h["n_edges"] for h in hist]
    phis = [h["phi_T"] for h in hist]
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_edges[::-1], phis[::-1], "b-o", linewidth=2)
    ax.set_xlabel("Number of edges (decreasing)")
    ax.set_ylabel("φ_T (profiled energy)")
    ax.set_title("Profiled Energy During Pruning")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phi_progression.png"), dpi=150)
    plt.close()


def plot_accuracy_comparison(all_best, output_dir):
    labels, data = zip(*sorted(all_best.items()))
    _, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["steelblue", "salmon"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, pts in enumerate(data):
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(pts))
        ax.scatter([i + 1 + j for j in jitter], pts, color="black", s=30, zorder=5, alpha=0.7)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Leverage Reduction vs Random Pruning")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: ExperimentConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    print("Loading Fashion-MNIST (patches)...")
    patches_tr, y_tr, patches_val, y_val, patches_te, y_te = load_data(cfg)

    variables = make_variables(cfg)
    candidates = make_candidates(cfg)
    print(f"Variables: {len(variables)}  Candidates: {len(candidates)}")

    # Build supergraph
    key, gk = jax.random.split(key)
    supergraph = ppc.build_supergraph(
        ppc.Graph(variables=variables, transforms=[], energies=[]),
        candidates, gk,
    )
    print(f"Supergraph: {len(supergraph.transforms)} edges")

    # Search batch from validation
    key, bk = jax.random.split(key)
    batch_idx = jax.random.permutation(bk, y_val.shape[0])[:min(256, y_val.shape[0])]
    search_clamps = make_clamps(patches_val, y_val, batch_idx, cfg.n_patches)

    # --- Leverage reduction ---
    reduce_results = []
    for trial in range(cfg.n_reduce_trials):
        print(f"\n{'='*60}")
        print(f" LEVERAGE REDUCTION trial {trial+1}/{cfg.n_reduce_trials}")
        print(f"{'='*60}")
        key, sk = jax.random.split(key)
        reduced, diag = ppc.reduce(supergraph, search_clamps, cfg.search, sk)
        edges = [t.id for t in reduced.transforms]
        print(f"  Result: {len(edges)} edges retained")
        for e in edges:
            lev = diag["final_leverage"].get(e, 0)
            print(f"    {e}  (leverage={lev:.4f})")
        reduce_results.append({"graph": reduced, "diagnostics": diag, "edges": edges})

    # --- Random pruning baseline (same number of edges) ---
    n_keep = len(reduce_results[0]["edges"]) if reduce_results else len(candidates) // 2
    random_results = []
    for trial in range(cfg.n_random_trials):
        print(f"\n{'='*60}")
        print(f" RANDOM PRUNING trial {trial+1}/{cfg.n_random_trials} (keeping {n_keep})")
        print(f"{'='*60}")
        key, sk = jax.random.split(key)
        reduced, diag = ppc.random_reduce(supergraph, search_clamps, n_keep, cfg.search, sk)
        edges = [t.id for t in reduced.transforms]
        print(f"  Result: {edges}")
        random_results.append({"graph": reduced, "diagnostics": diag, "edges": edges})

    # --- Train all ---
    trained = {"reduce": [], "random": []}
    for trial, r in enumerate(reduce_results):
        print(f"\n--- Training REDUCE trial {trial+1} ({len(r['edges'])} edges) ---")
        key, tk = jax.random.split(key)
        accs = train_and_evaluate(r["graph"], cfg, patches_tr, y_tr, patches_te, y_te, tk)
        trained["reduce"].append(accs)

    for trial, r in enumerate(random_results):
        print(f"\n--- Training RANDOM trial {trial+1} ({len(r['edges'])} edges) ---")
        key, tk = jax.random.split(key)
        accs = train_and_evaluate(r["graph"], cfg, patches_tr, y_tr, patches_te, y_te, tk)
        trained["random"].append(accs)

    # --- Plots ---
    if reduce_results:
        plot_leverage_distribution(reduce_results[0]["diagnostics"], cfg.output_dir)
        plot_phi_progression(reduce_results[0]["diagnostics"], cfg.output_dir)

    all_best = {
        "reduce": [max(a) if a else 0.0 for a in trained["reduce"]],
        "random": [max(a) if a else 0.0 for a in trained["random"]],
    }
    if all_best["reduce"] and all_best["random"]:
        plot_accuracy_comparison(all_best, cfg.output_dir)

    # --- Save ---
    output = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "reduce": [{"edges": r["edges"], "accs": trained["reduce"][i]}
                   for i, r in enumerate(reduce_results)],
        "random": [{"edges": r["edges"], "accs": trained["random"][i]}
                   for i, r in enumerate(random_results)],
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))

    # --- Summary ---
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for mode, bests in all_best.items():
        if bests:
            print(f"  {mode:15s}: {np.mean(bests):.4f} ± {np.std(bests):.4f}  (best={max(bests):.4f}, n={len(bests)})")
    print(f"\nOutputs: {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(ExperimentConfig)
    main(cfg)
