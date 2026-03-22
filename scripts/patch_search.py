"""Patch-only structure search on Fashion-MNIST.

No raw image variable — only 16 non-overlapping 7x7 patches.
Three-tier hierarchy: local → region → global → output.
6 transform types, ~98 candidates, cross-entropy output energy.
Key diagnostic: quick-eval accuracy after each search step.
"""

import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from keras.datasets import fashion_mnist as fmnist_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ppc
from scripts.search_utils import (
    plot_ablation_bars,
    plot_accuracy_boxplot,
    plot_eigenspectrum,
    plot_energy_gap,
    plot_quick_eval_curve,
    plot_score_vs_step,
    plot_search_comparison,
    plot_training_curves,
    plot_transform_types,
    print_architecture,
    save_results,
)


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
        conv_out = n_filters * (patch_h - 2) * (patch_w - 2)
        self.proj = eqx.nn.Linear(conv_out, out_dim, key=k2)

    def __call__(self, x):
        x = x.reshape(1, self.patch_h, self.patch_w)
        h = jax.nn.relu(self.conv(x))
        return self.proj(h.ravel())


class SafeLowRankLinear(eqx.Module):
    A: jax.Array
    B: jax.Array
    bias: jax.Array

    def __init__(self, in_dim, out_dim, rank=16, *, key):
        k1, k2 = jax.random.split(key)
        scale = 1.0 / jnp.sqrt(max(in_dim, out_dim))
        self.A = jax.random.normal(k1, (out_dim, rank)) * scale
        self.B = jax.random.normal(k2, (rank, in_dim)) * scale
        self.bias = jnp.zeros(out_dim)

    def __call__(self, x):
        return self.A @ (self.B @ x) + self.bias


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PatchSearchConfig:
    seed: int = 0
    batch_size: int = 256
    # Patch layout
    patch_h: int = 7
    patch_w: int = 7
    n_patches: int = 16
    # Variable dimensions
    local_dim: int = 48
    region_dim: int = 96
    n_regions: int = 4
    global_dim: int = 128
    n_globals: int = 2
    mlp_hidden: int = 64
    low_rank: int = 16
    # Search
    max_steps: int = 15
    epsilon_r: float = 0.0005
    eta: float = 0.05
    T: int = 20
    infer_iters: int = 30
    infer_lr: float = 0.05
    # Training
    train_epochs: int = 50
    quick_eval_epochs: int = 5
    quick_eval_samples: int = 3000
    train_lr: float = 1e-4
    train_batch_size: int = 64
    eval_every: int = 10
    # Experiment
    n_residual_trials: int = 3
    n_random_trials: int = 7
    val_size: int = 5000
    output_dir: str = "outputs/patch_search"


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    src: str
    tgt: str
    in_dim: int
    out_dim: int
    transform_type: str
    output_edge: bool = False  # True if tgt is "output"


# Patch-to-region spatial mapping
PATCH_TO_REGION = {
    0: 0, 1: 0, 4: 0, 5: 0,    # top-left
    2: 1, 3: 1, 6: 1, 7: 1,    # top-right
    8: 2, 9: 2, 12: 2, 13: 2,  # bottom-left
    10: 3, 11: 3, 14: 3, 15: 3, # bottom-right
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def extract_patches_grid(images, ph=7, pw=7):
    """Extract 16 non-overlapping patches from 28x28 in 4x4 grid."""
    n = images.shape[0]
    patches = []
    for row in range(4):
        for col in range(4):
            r, c = row * ph, col * pw
            p = images[:, r : r + ph, c : c + pw].reshape(n, -1).astype("float32") / 255.0
            patches.append(p)
    return patches


def load_data(cfg: PatchSearchConfig):
    (x_tr, y_tr), (x_te, y_te) = fmnist_data.load_data()
    # Train/val split
    x_val, y_val = x_tr[-cfg.val_size :], y_tr[-cfg.val_size :]
    x_tr, y_tr = x_tr[: -cfg.val_size], y_tr[: -cfg.val_size]
    patches_tr = [jnp.array(p) for p in extract_patches_grid(x_tr, cfg.patch_h, cfg.patch_w)]
    patches_val = [jnp.array(p) for p in extract_patches_grid(x_val, cfg.patch_h, cfg.patch_w)]
    patches_te = [jnp.array(p) for p in extract_patches_grid(x_te, cfg.patch_h, cfg.patch_w)]
    return (
        patches_tr, jax.nn.one_hot(y_tr, 10),
        patches_val, jax.nn.one_hot(y_val, 10),
        patches_te, jax.nn.one_hot(y_te, 10),
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def make_variables(cfg: PatchSearchConfig) -> list[ppc.Variable]:
    patch_dim = cfg.patch_h * cfg.patch_w
    variables = [ppc.Variable("output", (10,))]
    for i in range(cfg.n_patches):
        variables.append(ppc.Variable(f"p_{i}", (patch_dim,)))
    for i in range(cfg.n_patches):
        variables.append(ppc.Variable(f"h_local_{i}", (cfg.local_dim,)))
    for i in range(cfg.n_regions):
        variables.append(ppc.Variable(f"h_region_{i}", (cfg.region_dim,)))
    for i in range(cfg.n_globals):
        variables.append(ppc.Variable(f"h_global_{i}", (cfg.global_dim,)))
    return variables


def make_candidate_pool(cfg: PatchSearchConfig) -> list[Candidate]:
    patch_dim = cfg.patch_h * cfg.patch_w
    candidates = []

    # p_i → h_local_i: Linear, MLP_relu, Conv2d
    for i in range(cfg.n_patches):
        for ttype in ["Linear", "MLPRelu", "Conv2d"]:
            candidates.append(Candidate(
                f"p_{i}->h_local_{i}[{ttype}]",
                f"p_{i}", f"h_local_{i}", patch_dim, cfg.local_dim, ttype,
            ))

    # h_local_i → h_region_j (spatial): Linear
    for pi, ri in PATCH_TO_REGION.items():
        candidates.append(Candidate(
            f"h_local_{pi}->h_region_{ri}[Linear]",
            f"h_local_{pi}", f"h_region_{ri}", cfg.local_dim, cfg.region_dim, "Linear",
        ))

    # h_region_i → h_global_j: Linear, MLPGelu
    for i in range(cfg.n_regions):
        for j in range(cfg.n_globals):
            for ttype in ["Linear", "MLPGelu"]:
                candidates.append(Candidate(
                    f"h_region_{i}->h_global_{j}[{ttype}]",
                    f"h_region_{i}", f"h_global_{j}", cfg.region_dim, cfg.global_dim, ttype,
                ))

    # h_global_i → h_global_j (i!=j): Linear, MLPGelu
    for i in range(cfg.n_globals):
        for j in range(cfg.n_globals):
            if i != j:
                for ttype in ["Linear", "MLPGelu"]:
                    candidates.append(Candidate(
                        f"h_global_{i}->h_global_{j}[{ttype}]",
                        f"h_global_{i}", f"h_global_{j}", cfg.global_dim, cfg.global_dim, ttype,
                    ))

    # h_region_i → output: Linear, MLPRelu
    for i in range(cfg.n_regions):
        for ttype in ["Linear", "MLPRelu"]:
            candidates.append(Candidate(
                f"h_region_{i}->output[{ttype}]",
                f"h_region_{i}", "output", cfg.region_dim, 10, ttype, output_edge=True,
            ))

    # h_global_j → output: Linear, MLPRelu, MLPGelu
    for j in range(cfg.n_globals):
        for ttype in ["Linear", "MLPRelu", "MLPGelu"]:
            candidates.append(Candidate(
                f"h_global_{j}->output[{ttype}]",
                f"h_global_{j}", "output", cfg.global_dim, 10, ttype, output_edge=True,
            ))

    return candidates


def instantiate_candidate(
    cand: Candidate, edge_idx: int, cfg: PatchSearchConfig, key: jax.Array
) -> tuple[ppc.Transform, ppc.Energy]:
    tid = f"t_e{edge_idx}_{cand.name}"
    tt = cand.transform_type
    if tt == "Linear":
        module = eqx.nn.Linear(cand.in_dim, cand.out_dim, key=key)
    elif tt == "MLPRelu":
        module = MLPRelu(cand.in_dim, cfg.mlp_hidden, cand.out_dim, key=key)
    elif tt == "MLPGelu":
        module = MLPGelu(cand.in_dim, cfg.mlp_hidden, cand.out_dim, key=key)
    elif tt == "Conv2d":
        module = Conv2dBlock(cfg.patch_h, cfg.patch_w, cand.out_dim, key=key)
    elif tt == "SafeLowRank":
        module = SafeLowRankLinear(cand.in_dim, cand.out_dim, cfg.low_rank, key=key)
    else:
        raise ValueError(f"Unknown transform type: {tt}")

    transform = ppc.Transform(tid, module, src=cand.src, tgt=cand.tgt)
    # Cross-entropy for output edges, MSE otherwise
    efn = ppc.cross_entropy_energy if cand.output_edge else ppc.mse_energy
    energy = ppc.Energy(efn, args=[tid, cand.tgt])
    return transform, energy


def make_initial_graph(
    variables: list[ppc.Variable], cfg: PatchSearchConfig, key: jax.Array
) -> tuple[ppc.Graph, int]:
    """Bootstrap: p_0 →[Linear]→ h_local_0 →[Linear]→ output."""
    patch_dim = cfg.patch_h * cfg.patch_w
    k1, k2 = jax.random.split(key)
    t0 = ppc.Transform(
        "t_e0_p_0->h_local_0[Linear]",
        eqx.nn.Linear(patch_dim, cfg.local_dim, key=k1),
        src="p_0", tgt="h_local_0",
    )
    t1 = ppc.Transform(
        "t_e1_h_local_0->output[Linear]",
        eqx.nn.Linear(cfg.local_dim, 10, key=k2),
        src="h_local_0", tgt="output",
    )
    e0 = ppc.Energy(ppc.mse_energy, args=["t_e0_p_0->h_local_0[Linear]", "h_local_0"])
    e1 = ppc.Energy(ppc.cross_entropy_energy, args=["t_e1_h_local_0->output[Linear]", "output"])
    return ppc.Graph(variables=variables, transforms=[t0, t1], energies=[e0, e1]), 2


def make_clamps(patches, y, idx, cfg):
    """Build clamps dict from patches and labels at given indices."""
    clamps = {"output": y[idx]}
    for i in range(cfg.n_patches):
        clamps[f"p_{i}"] = patches[i][idx]
    return clamps


def make_clamps_nolabel(patches, idx, cfg):
    """Clamps without output (for evaluation)."""
    clamps = {}
    for i in range(cfg.n_patches):
        clamps[f"p_{i}"] = patches[i][idx]
    return clamps


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_candidates(
    graph: ppc.Graph,
    state: ppc.State,
    decomp: dict,
    candidates: list[Candidate],
    clamps: dict[str, jax.Array],
    edge_idx: int,
    cfg: PatchSearchConfig,
    key: jax.Array,
) -> list[tuple[int, float]]:
    """Score candidates: Woodbury matching pursuit + φ_T for exploration.

    Two tiers per the paper:
    1. Woodbury gain (eq. 8): one forward pass per candidate. Measures alignment
       of new Jacobian directions with unresolved residual. This is the matching
       pursuit signal — works for candidates touching already-connected variables.
    2. For candidates with zero Woodbury gain (touching disconnected variables),
       use φ_T from decompose on the expanded graph — measures whether the new
       edge improves the precision matrix geometry.
    """
    infer_opt = optax.adam(cfg.infer_lr)
    keys = jax.random.split(key, len(candidates))

    # Tier 1: Woodbury gain for all candidates
    woodbury_scores = []
    for i, cand in enumerate(candidates):
        t, e = instantiate_candidate(cand, edge_idx + i, cfg, keys[i])
        try:
            gain = ppc.woodbury_gain(graph, state, t, decomp)
        except Exception:
            gain = 0.0
        woodbury_scores.append((i, gain))

    # Separate connected (gain > 0) from disconnected (gain == 0)
    connected = [(i, g) for i, g in woodbury_scores if g > 1e-10]
    disconnected = [(i, g) for i, g in woodbury_scores if g <= 1e-10]

    # Tier 2: for disconnected candidates, evaluate φ_T on expanded graph
    # (this is the exploration signal — which new region should we reach?)
    base_phi = float(jnp.mean(decomp["phi_T_predicted"]))
    explored = []
    for i, _ in disconnected:
        cand = candidates[i]
        k1, k2 = jax.random.split(keys[i])
        t, e = instantiate_candidate(cand, edge_idx + i, cfg, k1)
        expanded = ppc.expand(graph, new_transforms=[t], new_energies=[e])
        try:
            exp_state = ppc.init(expanded, clamps, key=k2)
            exp_state = ppc.infer(expanded, exp_state, optimizer=infer_opt, iters=cfg.infer_iters)
            d_new = ppc.decompose(expanded, exp_state, cfg.eta, cfg.T)
            new_phi = float(jnp.mean(d_new["phi_T_predicted"]))
            # Use φ_T reduction as gain; scale to be comparable with Woodbury
            gain = max(0.0, base_phi - new_phi)
        except Exception:
            gain = 0.0
        explored.append((i, gain))

    # Merge: all candidates with their best score
    all_scores = connected + explored
    all_scores.sort(key=lambda x: x[1], reverse=True)
    return all_scores


# ---------------------------------------------------------------------------
# Quick eval
# ---------------------------------------------------------------------------


def quick_eval(
    graph: ppc.Graph,
    cfg: PatchSearchConfig,
    patches_tr, y_tr,
    patches_val, y_val,
    key: jax.Array,
) -> float:
    """Train copy of graph briefly, return val accuracy. Does NOT modify input graph."""
    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    g = graph
    opt_state = train_opt.init(eqx.filter(g, eqx.is_array))
    bs = cfg.train_batch_size
    n_tr = min(y_tr.shape[0], cfg.quick_eval_samples)

    train_step = eqx.filter_jit(lambda g, os, c, k: _train_step(g, os, c, k, infer_opt, train_opt, cfg))

    for epoch in range(cfg.quick_eval_epochs):
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, y_tr.shape[0])[:n_tr]
        for i in range(0, n_tr - bs + 1, bs):
            idx = perm[i : i + bs]
            key, sk = jax.random.split(key)
            clamps = make_clamps(patches_tr, y_tr, idx, cfg)
            g, opt_state, _ = train_step(g, opt_state, clamps, sk)

    # Eval on val
    n_val = min(y_val.shape[0], cfg.quick_eval_samples)
    correct, total = 0, 0
    eval_step = eqx.filter_jit(lambda g, c, k: _eval_step(g, c, k, infer_opt, cfg))
    for i in range(0, n_val - bs + 1, bs):
        key, ek = jax.random.split(key)
        idx = jnp.arange(i, i + bs)
        clamps_eval = make_clamps_nolabel(patches_val, idx, cfg)
        preds = eval_step(g, clamps_eval, ek)
        correct += int(jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y_val[i : i + bs], -1)))
        total += bs
    return correct / total if total > 0 else 0.0


def _train_step(g, opt_state, clamps, key, infer_opt, train_opt, cfg):
    state = ppc.init(g, clamps, key=key)
    state = ppc.infer(g, state, optimizer=infer_opt, iters=cfg.infer_iters)
    loss = ppc.energy(g, state)
    grads = ppc.param_grad(g, state)
    updates, opt_state = train_opt.update(
        eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(g, eqx.is_array)
    )
    g = eqx.apply_updates(g, updates)
    return g, opt_state, loss


def _eval_step(g, clamps, key, infer_opt, cfg):
    state = ppc.init(g, clamps, key=key)
    state = ppc.infer(g, state, optimizer=infer_opt, iters=cfg.infer_iters * 2)
    return ppc.variable(g, state, "output")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def run_search(
    cfg: PatchSearchConfig,
    variables: list[ppc.Variable],
    candidates: list[Candidate],
    patches_tr, y_tr,
    patches_val, y_val,
    key: jax.Array,
    mode: str = "residual",
) -> tuple[ppc.Graph, dict]:
    key, gk = jax.random.split(key)
    graph, edge_idx = make_initial_graph(variables, cfg, gk)
    remaining = list(range(len(candidates)))

    # Remove bootstrap candidates
    bootstrap_names = {"p_0->h_local_0[Linear]", "h_local_0->output[Linear]"}
    for i, c in enumerate(candidates):
        if c.name in bootstrap_names and i in remaining:
            remaining.remove(i)

    infer_opt = optax.adam(cfg.infer_lr)
    history = {
        "coverage_gap": [], "conditioning_penalty": [], "phi_T": [],
        "effective_rank": [], "eigenvalues": [],
        "energy_true": [], "energy_wrong": [], "energy_gap": [],
        "selected_edge": [], "gain": [],
        "quick_eval_acc": [],
        "top5": [],
    }

    for step in range(cfg.max_steps):
        if not remaining:
            print(f"  Step {step}: No more candidates")
            break

        # Sample batch from validation for decompose
        key, bk, ik = jax.random.split(key, 3)
        batch_idx = jax.random.permutation(bk, y_val.shape[0])[: cfg.batch_size]
        clamps = make_clamps(patches_val, y_val, batch_idx, cfg)
        state = ppc.init(graph, clamps, key=ik)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters)

        # Decompose
        decomp = ppc.decompose(graph, state, cfg.eta, cfg.T)
        cg = float(jnp.mean(decomp["coverage_gap"]))
        cp = float(jnp.mean(decomp["conditioning_penalty"]))
        phi = float(jnp.mean(decomp["phi_T_predicted"]))
        er = decomp["effective_rank"]
        eigs = np.array(decomp["eigenvalues"])

        # Energy gap
        e_true = float(ppc.energy(graph, state)) / cfg.batch_size
        key, wk, wik = jax.random.split(key, 3)
        wrong_perm = jax.random.permutation(wk, cfg.batch_size)
        clamps_wrong = {**clamps, "output": clamps["output"][wrong_perm]}
        s_wrong = ppc.init(graph, clamps_wrong, key=wik)
        s_wrong = ppc.infer(graph, s_wrong, optimizer=infer_opt, iters=cfg.infer_iters)
        e_wrong = float(ppc.energy(graph, s_wrong)) / cfg.batch_size
        e_gap = e_wrong - e_true

        history["coverage_gap"].append(cg)
        history["conditioning_penalty"].append(cp)
        history["phi_T"].append(phi)
        history["effective_rank"].append(er)
        history["eigenvalues"].append(eigs)
        history["energy_true"].append(e_true)
        history["energy_wrong"].append(e_wrong)
        history["energy_gap"].append(e_gap)

        print(
            f"  [{mode}] Step {step}: cg={cg:.4f}  cp={cp:.4f}  "
            f"phi_T={phi:.4f}  e_gap={e_gap:.4f}  eff_rank={er}  "
            f"n_edges={len(graph.transforms)}"
        )

        if cg < cfg.epsilon_r:
            history["selected_edge"].append(None)
            history["gain"].append(0.0)
            history["quick_eval_acc"].append(history["quick_eval_acc"][-1] if history["quick_eval_acc"] else 0.0)
            history["top5"].append([])
            break

        # Score candidates
        pool = [candidates[i] for i in remaining]
        if mode == "residual":
            key, sk = jax.random.split(key)
            scored = score_candidates(graph, state, decomp, pool, clamps, edge_idx, cfg, sk)
            top5 = [(pool[s[0]].name, round(s[1], 6)) for s in scored[:5]]
            history["top5"].append(top5)
            print(f"    Top-5: {top5}")
            best_pool_idx, best_gain = scored[0][0], scored[0][1]
        else:
            key, rk = jax.random.split(key)
            best_pool_idx = int(jax.random.randint(rk, (), 0, len(pool)))
            best_gain = 0.0
            history["top5"].append([])

        selected_cand = pool[best_pool_idx]
        history["selected_edge"].append(selected_cand.name)
        history["gain"].append(best_gain)

        # Expand
        key, ek = jax.random.split(key)
        t, e = instantiate_candidate(selected_cand, edge_idx, cfg, ek)
        graph = ppc.expand(graph, new_transforms=[t], new_energies=[e])
        edge_idx += 1
        remaining.remove(remaining[best_pool_idx])

        # Quick eval
        key, qk = jax.random.split(key)
        qe_acc = quick_eval(graph, cfg, patches_tr, y_tr, patches_val, y_val, qk)
        history["quick_eval_acc"].append(qe_acc)

        print(f"    -> Selected: {selected_cand.name} (gain={best_gain:.6f}, quick_acc={qe_acc:.4f})")

    return graph, history


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------


def train_and_evaluate(
    graph: ppc.Graph,
    cfg: PatchSearchConfig,
    patches_tr, y_tr,
    patches_te, y_te,
    key: jax.Array,
) -> list[float]:
    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))
    bs = cfg.train_batch_size

    train_step = eqx.filter_jit(lambda g, os, c, k: _train_step(g, os, c, k, infer_opt, train_opt, cfg))
    eval_step = eqx.filter_jit(lambda g, c, k: _eval_step(g, c, k, infer_opt, cfg))

    # Warmup
    key, sk = jax.random.split(key)
    c = make_clamps(patches_tr, y_tr, jnp.arange(bs), cfg)
    graph, opt_state, _ = train_step(graph, opt_state, c, sk)

    accs = []
    for epoch in range(1, cfg.train_epochs + 1):
        t0 = time.perf_counter()
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, y_tr.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, y_tr.shape[0] - bs + 1, bs):
            idx = perm[i : i + bs]
            key, sk = jax.random.split(key)
            c = make_clamps(patches_tr, y_tr, idx, cfg)
            graph, opt_state, loss = train_step(graph, opt_state, c, sk)
            total_loss += float(loss)
            n += 1
        dt = time.perf_counter() - t0

        if epoch % cfg.eval_every == 0 or epoch == cfg.train_epochs:
            correct, total = 0, 0
            for i in range(0, y_te.shape[0] - bs + 1, bs):
                key, ek2 = jax.random.split(key)
                idx = jnp.arange(i, i + bs)
                c_eval = make_clamps_nolabel(patches_te, idx, cfg)
                preds = eval_step(graph, c_eval, ek2)
                correct += int(jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y_te[i : i + bs], -1)))
                total += bs
            acc = correct / total if total > 0 else 0.0
            accs.append(acc)
            print(f"    Epoch {epoch}/{cfg.train_epochs}  loss={total_loss / n:.2f}  acc={acc:.4f}  ({dt:.1f}s)")
        else:
            print(f"    Epoch {epoch}/{cfg.train_epochs}  loss={total_loss / n:.2f}  ({dt:.1f}s)")
    return accs


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def ablation_study(
    graph: ppc.Graph,
    cfg: PatchSearchConfig,
    patches_tr, y_tr,
    patches_val, y_val,
    key: jax.Array,
    base_acc: float,
) -> list[dict]:
    """Remove each non-bootstrap edge, quick-eval, measure accuracy drop."""
    bootstrap_ids = {graph.transforms[0].id, graph.transforms[1].id}
    results = []
    for i, t in enumerate(graph.transforms):
        if t.id in bootstrap_ids:
            continue
        rem_transforms = [tr for j, tr in enumerate(graph.transforms) if j != i]
        rem_energies = [e for e in graph.energies if t.id not in e.args]
        try:
            ablated = ppc.Graph(
                variables=list(graph.variables),
                transforms=rem_transforms,
                energies=rem_energies,
            )
            key, ak = jax.random.split(key)
            acc = quick_eval(ablated, cfg, patches_tr, y_tr, patches_val, y_val, ak)
        except Exception:
            acc = 0.0
        drop = base_acc - acc
        results.append({"edge": t.id, "acc_without": acc, "acc_drop": drop})
        print(f"    Ablation: remove {t.id.split('_', 2)[-1]} -> acc={acc:.4f} (drop={drop:+.4f})")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: PatchSearchConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    print("Loading Fashion-MNIST (patch-only)...")
    patches_tr, y_tr, patches_val, y_val, patches_te, y_te = load_data(cfg)

    variables = make_variables(cfg)
    candidates = make_candidate_pool(cfg)
    print(f"Variables: {len(variables)} ({sum(1 for v in variables if v.name.startswith('p_'))} patches, "
          f"{sum(1 for v in variables if 'local' in v.name)} local, "
          f"{sum(1 for v in variables if 'region' in v.name)} region, "
          f"{sum(1 for v in variables if 'global' in v.name)} global, 1 output)")
    print(f"Candidate pool: {len(candidates)} edges")
    print(f"  Transform types: {sorted(set(c.transform_type for c in candidates))}")

    # --- Residual-driven searches ---
    res_results = []
    for trial in range(cfg.n_residual_trials):
        print(f"\n{'='*60}")
        print(f"=== Residual-Driven Search (trial {trial + 1}/{cfg.n_residual_trials}) ===")
        print(f"{'='*60}")
        key, rk = jax.random.split(key)
        g, h = run_search(cfg, variables, candidates, patches_tr, y_tr, patches_val, y_val, rk, "residual")
        print_architecture(g.transforms, f"Residual trial {trial+1}")
        res_results.append((g, h))

    # --- Random searches ---
    rand_results = []
    for trial in range(cfg.n_random_trials):
        print(f"\n{'='*60}")
        print(f"=== Random Search (trial {trial + 1}/{cfg.n_random_trials}) ===")
        print(f"{'='*60}")
        key, rk = jax.random.split(key)
        g, h = run_search(cfg, variables, candidates, patches_tr, y_tr, patches_val, y_val, rk, "random")
        rand_results.append((g, h))

    # --- Plots: search phase ---
    print("\n=== Generating search-phase plots ===")
    res_histories = [h for _, h in res_results]
    rand_histories = [h for _, h in rand_results]
    plot_quick_eval_curve(
        res_histories, rand_histories,
        os.path.join(cfg.output_dir, "quick_eval_curve.png"),
    )
    # Use first residual trial for comparison plots
    plot_search_comparison(res_histories[0], rand_histories, os.path.join(cfg.output_dir, "search_comparison.png"))
    plot_score_vs_step(res_histories[0], rand_histories, os.path.join(cfg.output_dir, "score_vs_step.png"))
    plot_energy_gap(res_histories[0], rand_histories, os.path.join(cfg.output_dir, "energy_gap.png"))
    if res_histories[0]["eigenvalues"]:
        plot_eigenspectrum(res_histories[0]["eigenvalues"], os.path.join(cfg.output_dir, "eigenspectrum.png"))
    plot_transform_types(
        [t.id for t in res_results[0][0].transforms],
        [[t.id for t in g.transforms] for g, _ in rand_results],
        os.path.join(cfg.output_dir, "transform_types.png"),
    )

    # --- Train all architectures ---
    res_accs = []
    for trial, (g, _) in enumerate(res_results):
        print(f"\n=== Training residual architecture (trial {trial + 1}) ===")
        key, tk = jax.random.split(key)
        accs = train_and_evaluate(g, cfg, patches_tr, y_tr, patches_te, y_te, tk)
        res_accs.append(accs)

    rand_accs = []
    for trial, (g, _) in enumerate(rand_results):
        print(f"\n=== Training random architecture (trial {trial + 1}) ===")
        key, tk = jax.random.split(key)
        accs = train_and_evaluate(g, cfg, patches_tr, y_tr, patches_te, y_te, tk)
        rand_accs.append(accs)

    # --- Plots: training phase ---
    if res_accs and res_accs[0]:
        plot_training_curves(res_accs[0], rand_accs, os.path.join(cfg.output_dir, "training_curves.png"))
    res_best = [max(a) if a else 0.0 for a in res_accs]
    rand_best = [max(a) if a else 0.0 for a in rand_accs]
    if res_best and rand_best:
        plot_accuracy_boxplot(res_best, rand_best, os.path.join(cfg.output_dir, "accuracy_boxplot.png"))

    # --- Ablation on best residual ---
    best_idx = max(range(len(res_accs)), key=lambda i: max(res_accs[i]) if res_accs[i] else 0)
    best_acc = max(res_accs[best_idx]) if res_accs[best_idx] else 0.0
    print(f"\n=== Ablation Study (best residual trial {best_idx+1}, acc={best_acc:.4f}) ===")
    key, ak = jax.random.split(key)
    ablation = ablation_study(
        res_results[best_idx][0], cfg, patches_tr, y_tr, patches_val, y_val, ak, best_acc
    )
    if ablation:
        plot_ablation_bars(ablation, best_acc, os.path.join(cfg.output_dir, "ablation.png"))

    # --- Save results ---
    results = {
        "config": vars(cfg),
        "residual_driven": [
            {
                "history": {k: v for k, v in h.items() if k != "eigenvalues"},
                "edges": [t.id for t in g.transforms],
                "final_accs": a,
            }
            for (g, h), a in zip(res_results, res_accs)
        ],
        "random": [
            {
                "history": {k: v for k, v in h.items() if k != "eigenvalues"},
                "edges": [t.id for t in g.transforms],
                "final_accs": a,
            }
            for (g, h), a in zip(rand_results, rand_accs)
        ],
        "ablation": ablation,
    }
    save_results(results, cfg.output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nResidual-driven ({cfg.n_residual_trials} trials):")
    for i, (accs, (g, h)) in enumerate(zip(res_accs, res_results)):
        best = max(accs) if accs else 0.0
        n_edges = len(g.transforms)
        qe = h.get("quick_eval_acc", [])
        qe_final = qe[-1] if qe else 0.0
        print(f"  Trial {i+1}: {n_edges} edges, best_acc={best:.4f}, final_quick_eval={qe_final:.4f}")
    print(f"  Mean best accuracy: {np.mean(res_best):.4f} +/- {np.std(res_best):.4f}")

    print(f"\nRandom ({cfg.n_random_trials} trials):")
    for i, (accs, (g, h)) in enumerate(zip(rand_accs, rand_results)):
        best = max(accs) if accs else 0.0
        print(f"  Trial {i+1}: {len(g.transforms)} edges, best_acc={best:.4f}")
    print(f"  Mean best accuracy: {np.mean(rand_best):.4f} +/- {np.std(rand_best):.4f}")

    gap = np.mean(res_best) - np.mean(rand_best)
    print(f"\n  ACCURACY GAP: {gap:+.4f} (residual - random)")
    print(f"\nOutputs saved to {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(PatchSearchConfig)
    main(cfg)
