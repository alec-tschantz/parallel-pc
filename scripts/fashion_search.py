"""Multi-scale structure search on Fashion-MNIST with diverse transform types.

Compares greedy residual-driven expansion against random search baseline.
The search space includes Linear, MLP, Conv1d, and LowRank transforms
connecting multi-scale variables (patches, local features, global features).
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
    plot_eigenspectrum,
    plot_energy_gap,
    plot_score_vs_step,
    plot_search_comparison,
    plot_training_curves,
    plot_transform_types,
    save_results,
)


# ---------------------------------------------------------------------------
# Custom transform modules
# ---------------------------------------------------------------------------


class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, out_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, out_dim, key=k2)

    def __call__(self, x):
        return self.linear2(jax.nn.relu(self.linear1(x)))


class Conv1dBlock(eqx.Module):
    conv: eqx.nn.Conv1d
    proj: eqx.nn.Linear

    def __init__(self, in_dim, out_dim, kernel_size=5, *, key):
        k1, k2 = jax.random.split(key)
        out_channels = 4
        conv_out_len = in_dim - kernel_size + 1
        self.conv = eqx.nn.Conv1d(1, out_channels, kernel_size, key=k1)
        self.proj = eqx.nn.Linear(out_channels * conv_out_len, out_dim, key=k2)

    def __call__(self, x):
        h = self.conv(x[None, :])  # (1, in_dim) -> (out_channels, conv_out_len)
        h = jax.nn.relu(h)
        return self.proj(h.ravel())


class LowRankLinear(eqx.Module):
    A: jax.Array
    B: jax.Array
    bias: jax.Array

    def __init__(self, in_dim, out_dim, rank=8, *, key):
        k1, k2 = jax.random.split(key)
        scale = 1.0 / jnp.sqrt(rank)
        self.A = jax.random.normal(k1, (out_dim, rank)) * scale
        self.B = jax.random.normal(k2, (rank, in_dim)) * scale
        self.bias = jnp.zeros(out_dim)

    def __call__(self, x):
        return self.A @ (self.B @ x) + self.bias


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FashionSearchConfig:
    seed: int = 0
    batch_size: int = 128
    # Variable dimensions
    patch_dim: int = 49  # 7x7
    n_patches: int = 4
    local_dim: int = 32
    n_local: int = 4
    global_dim: int = 64
    n_global: int = 3
    # Search
    max_steps: int = 12
    epsilon_r: float = 0.001
    eta: float = 0.05
    T: int = 20
    infer_iters: int = 20
    infer_lr: float = 0.05
    # Transform params
    mlp_hidden: int = 32
    low_rank: int = 8
    conv_kernel: int = 5
    # Training
    train_epochs: int = 20
    train_lr: float = 5e-5
    train_batch_size: int = 64
    eval_every: int = 5
    # Baseline
    n_random_trials: int = 7
    output_dir: str = "outputs/fashion_search"


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
    transform_type: str  # "Linear", "MLP", "Conv1d", "LowRank"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def extract_patches(images):
    """Extract 4 non-overlapping 7x7 patches from 28x28 images (corners)."""
    patches = []
    for r in (0, 21):
        for c in (0, 21):
            p = images[:, r : r + 7, c : c + 7].reshape(-1, 49).astype("float32") / 255.0
            patches.append(p)
    return patches


def load_fashion_mnist():
    (x_tr, y_tr), (x_te, y_te) = fmnist_data.load_data()
    x_tr_flat = jnp.array(x_tr.reshape(-1, 784).astype("float32") / 255.0)
    x_te_flat = jnp.array(x_te.reshape(-1, 784).astype("float32") / 255.0)
    patches_tr = [jnp.array(p) for p in extract_patches(x_tr)]
    patches_te = [jnp.array(p) for p in extract_patches(x_te)]
    return (
        x_tr_flat, jax.nn.one_hot(y_tr, 10),
        x_te_flat, jax.nn.one_hot(y_te, 10),
        patches_tr, patches_te,
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def make_variables(cfg: FashionSearchConfig) -> list[ppc.Variable]:
    variables = [
        ppc.Variable("image", (784,)),
        ppc.Variable("output", (10,)),
    ]
    for i in range(cfg.n_patches):
        variables.append(ppc.Variable(f"patch_{i}", (cfg.patch_dim,)))
    for i in range(cfg.n_local):
        variables.append(ppc.Variable(f"h_local_{i}", (cfg.local_dim,)))
    for i in range(cfg.n_global):
        variables.append(ppc.Variable(f"h_global_{i}", (cfg.global_dim,)))
    return variables


def make_candidate_pool(variables: list[ppc.Variable], cfg: FashionSearchConfig) -> list[Candidate]:
    var_map = {v.name: v for v in variables}
    candidates = []

    # patch_i -> h_local_i with {Linear, MLP, Conv1d}
    for i in range(cfg.n_patches):
        src, tgt = f"patch_{i}", f"h_local_{i}"
        for ttype in ["Linear", "MLP", "Conv1d"]:
            candidates.append(Candidate(
                f"{src}->{tgt}[{ttype}]", src, tgt,
                var_map[src].shape[0], var_map[tgt].shape[0], ttype,
            ))

    # image -> h_global_j with {Linear, MLP, LowRank}
    for j in range(cfg.n_global):
        src, tgt = "image", f"h_global_{j}"
        for ttype in ["Linear", "MLP", "LowRank"]:
            candidates.append(Candidate(
                f"{src}->{tgt}[{ttype}]", src, tgt,
                var_map[src].shape[0], var_map[tgt].shape[0], ttype,
            ))

    # h_local_i -> h_global_j with {Linear, MLP}
    for i in range(cfg.n_local):
        for j in range(cfg.n_global):
            src, tgt = f"h_local_{i}", f"h_global_{j}"
            for ttype in ["Linear", "MLP"]:
                candidates.append(Candidate(
                    f"{src}->{tgt}[{ttype}]", src, tgt,
                    var_map[src].shape[0], var_map[tgt].shape[0], ttype,
                ))

    # h_global_i -> h_global_j (i != j) with {Linear, MLP}
    for i in range(cfg.n_global):
        for j in range(cfg.n_global):
            if i != j:
                src, tgt = f"h_global_{i}", f"h_global_{j}"
                for ttype in ["Linear", "MLP"]:
                    candidates.append(Candidate(
                        f"{src}->{tgt}[{ttype}]", src, tgt,
                        var_map[src].shape[0], var_map[tgt].shape[0], ttype,
                    ))

    # h_local_i -> output with {Linear}
    for i in range(cfg.n_local):
        src = f"h_local_{i}"
        candidates.append(Candidate(
            f"{src}->output[Linear]", src, "output",
            var_map[src].shape[0], 10, "Linear",
        ))

    # h_global_j -> output with {Linear, MLP}
    for j in range(cfg.n_global):
        src = f"h_global_{j}"
        for ttype in ["Linear", "MLP"]:
            candidates.append(Candidate(
                f"{src}->output[{ttype}]", src, "output",
                var_map[src].shape[0], 10, ttype,
            ))

    # image -> output with {Linear, LowRank}
    for ttype in ["Linear", "LowRank"]:
        candidates.append(Candidate(
            f"image->output[{ttype}]", "image", "output", 784, 10, ttype,
        ))

    return candidates


def instantiate_candidate(
    candidate: Candidate, edge_idx: int, cfg: FashionSearchConfig, key: jax.Array
) -> tuple[ppc.Transform, ppc.Energy]:
    tid = f"t_e{edge_idx}_{candidate.name}"
    if candidate.transform_type == "Linear":
        module = eqx.nn.Linear(candidate.in_dim, candidate.out_dim, key=key)
    elif candidate.transform_type == "MLP":
        module = MLP(candidate.in_dim, cfg.mlp_hidden, candidate.out_dim, key=key)
    elif candidate.transform_type == "Conv1d":
        module = Conv1dBlock(candidate.in_dim, candidate.out_dim, cfg.conv_kernel, key=key)
    elif candidate.transform_type == "LowRank":
        module = LowRankLinear(candidate.in_dim, candidate.out_dim, cfg.low_rank, key=key)
    else:
        raise ValueError(f"Unknown: {candidate.transform_type}")
    transform = ppc.Transform(tid, module, src=candidate.src, tgt=candidate.tgt)
    energy = ppc.Energy(ppc.mse_energy, args=[tid, candidate.tgt])
    return transform, energy


def make_initial_graph(
    variables: list[ppc.Variable], cfg: FashionSearchConfig, key: jax.Array
) -> tuple[ppc.Graph, int]:
    """Bootstrap: image -> h_global_0 -> output (both Linear)."""
    k1, k2 = jax.random.split(key)
    t0 = ppc.Transform(
        "t_e0_image->h_global_0[Linear]",
        eqx.nn.Linear(784, cfg.global_dim, key=k1),
        src="image", tgt="h_global_0",
    )
    t1 = ppc.Transform(
        "t_e1_h_global_0->output[Linear]",
        eqx.nn.Linear(cfg.global_dim, 10, key=k2),
        src="h_global_0", tgt="output",
    )
    e0 = ppc.Energy(ppc.mse_energy, args=["t_e0_image->h_global_0[Linear]", "h_global_0"])
    e1 = ppc.Energy(ppc.mse_energy, args=["t_e1_h_global_0->output[Linear]", "output"])
    graph = ppc.Graph(variables=variables, transforms=[t0, t1], energies=[e0, e1])
    return graph, 2


def make_clamps(x_batch, y_batch, patches_batch, n_patches):
    clamps = {"image": x_batch, "output": y_batch}
    for i in range(n_patches):
        clamps[f"patch_{i}"] = patches_batch[i]
    return clamps


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def score_candidates(
    graph: ppc.Graph,
    candidates: list[Candidate],
    clamps: dict[str, jax.Array],
    edge_idx: int,
    cfg: FashionSearchConfig,
    key: jax.Array,
) -> list[tuple[int, float]]:
    """Score ALL candidates by energy gap (correct vs wrong labels)."""
    infer_opt = optax.adam(cfg.infer_lr)
    batch_size = clamps["image"].shape[0]
    keys = jax.random.split(key, len(candidates) + 1)
    perm_key = keys[-1]
    keys = keys[:-1]

    wrong_perm = jax.random.permutation(perm_key, batch_size)
    clamps_wrong = {**clamps, "output": clamps["output"][wrong_perm]}

    gains = []
    for i, cand in enumerate(candidates):
        k1, k2, k3 = jax.random.split(keys[i], 3)
        t, e = instantiate_candidate(cand, edge_idx + i, cfg, k1)
        expanded = ppc.expand(graph, new_transforms=[t], new_energies=[e])
        try:
            s_true = ppc.init(expanded, clamps, key=k2)
            s_true = ppc.infer(expanded, s_true, optimizer=infer_opt, iters=cfg.infer_iters)
            e_true = float(ppc.energy(expanded, s_true)) / batch_size

            s_wrong = ppc.init(expanded, clamps_wrong, key=k3)
            s_wrong = ppc.infer(expanded, s_wrong, optimizer=infer_opt, iters=cfg.infer_iters)
            e_wrong = float(ppc.energy(expanded, s_wrong)) / batch_size

            gain = e_wrong - e_true
        except Exception:
            gain = -1e9
        gains.append((i, gain))

    gains.sort(key=lambda x: x[1], reverse=True)
    return gains


def run_search(
    cfg: FashionSearchConfig,
    variables: list[ppc.Variable],
    candidates: list[Candidate],
    x_batch: jax.Array,
    y_batch: jax.Array,
    patches_batch: list[jax.Array],
    key: jax.Array,
    mode: str = "residual",
) -> tuple[ppc.Graph, dict]:
    key, gk = jax.random.split(key)
    graph, edge_idx = make_initial_graph(variables, cfg, gk)
    remaining = list(range(len(candidates)))

    bootstrap_names = {"image->h_global_0[Linear]", "h_global_0->output[Linear]"}
    for i, c in enumerate(candidates):
        if c.name in bootstrap_names and i in remaining:
            remaining.remove(i)

    infer_opt = optax.adam(cfg.infer_lr)
    history = {
        "coverage_gap": [], "conditioning_penalty": [], "phi_T": [],
        "effective_rank": [], "eigenvalues": [],
        "energy_true": [], "energy_wrong": [], "energy_gap": [],
        "selected_edge": [], "gain": [],
    }

    for step in range(cfg.max_steps):
        if not remaining:
            print(f"  Step {step}: No more candidates")
            break

        key, ik = jax.random.split(key)
        clamps = make_clamps(x_batch, y_batch, patches_batch, cfg.n_patches)
        state = ppc.init(graph, clamps, key=ik)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters)

        decomp = ppc.decompose(graph, state, cfg.eta, cfg.T)
        cg = float(jnp.mean(decomp["coverage_gap"]))
        cp = float(jnp.mean(decomp["conditioning_penalty"]))
        phi = float(jnp.mean(decomp["phi_T_predicted"]))
        er = decomp["effective_rank"]
        eigs = np.array(decomp["eigenvalues"])

        # Energy gap
        e_true = float(ppc.energy(graph, state)) / x_batch.shape[0]
        key, wk, wik = jax.random.split(key, 3)
        wrong_perm = jax.random.permutation(wk, x_batch.shape[0])
        clamps_wrong = {**clamps, "output": y_batch[wrong_perm]}
        s_wrong = ppc.init(graph, clamps_wrong, key=wik)
        s_wrong = ppc.infer(graph, s_wrong, optimizer=infer_opt, iters=cfg.infer_iters)
        e_wrong = float(ppc.energy(graph, s_wrong)) / x_batch.shape[0]
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
            break

        pool = [candidates[i] for i in remaining]
        if mode == "residual":
            key, sk = jax.random.split(key)
            scored = score_candidates(graph, pool, clamps, edge_idx, cfg, sk)
            best_pool_idx, best_gain = scored[0]
            selected_remaining_idx = best_pool_idx
        else:
            key, rk = jax.random.split(key)
            selected_remaining_idx = int(jax.random.randint(rk, (), 0, len(pool)))
            best_gain = 0.0

        selected_cand = pool[selected_remaining_idx]
        history["selected_edge"].append(selected_cand.name)
        history["gain"].append(best_gain)

        key, ek = jax.random.split(key)
        t, e = instantiate_candidate(selected_cand, edge_idx, cfg, ek)
        graph = ppc.expand(graph, new_transforms=[t], new_energies=[e])
        edge_idx += 1
        remaining.remove(remaining[selected_remaining_idx])

        print(f"    -> Selected: {selected_cand.name} (gain={best_gain:.6f})")

    return graph, history


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_evaluate(
    graph: ppc.Graph,
    cfg: FashionSearchConfig,
    x_tr, y_tr, x_te, y_te,
    patches_tr, patches_te,
    key: jax.Array,
) -> list[float]:
    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))
    bs = cfg.train_batch_size
    np_ = cfg.n_patches

    def _make_clamps_train(images, labels, patches_list, idx):
        c = {"image": images, "output": labels}
        for i in range(np_):
            c[f"patch_{i}"] = patches_list[i][idx] if not isinstance(idx, slice) else patches_list[i][idx]
        return c

    def _make_clamps_eval(images, patches_list, idx):
        c = {"image": images}
        for i in range(np_):
            c[f"patch_{i}"] = patches_list[i][idx]
        return c

    # Can't jit with dict clamps directly — use a wrapper approach
    # We'll not jit train/eval for simplicity (search is the bottleneck, not training)
    def train_step(graph, opt_state, clamps, key):
        state = ppc.init(graph, clamps, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters)
        loss = ppc.energy(graph, state)
        grads = ppc.param_grad(graph, state)
        updates, opt_state = train_opt.update(
            eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(graph, eqx.is_array)
        )
        graph = eqx.apply_updates(graph, updates)
        return graph, opt_state, loss

    def eval_step(graph, clamps, key):
        state = ppc.init(graph, clamps, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters * 2)
        return ppc.variable(graph, state, "output")

    # Compile once via jit on the first call pattern
    train_step = eqx.filter_jit(train_step)
    eval_step = eqx.filter_jit(eval_step)

    # Warmup
    key, sk = jax.random.split(key)
    c = make_clamps(x_tr[:bs], y_tr[:bs], [p[:bs] for p in patches_tr], np_)
    graph, opt_state, _ = train_step(graph, opt_state, c, sk)

    accs = []
    for epoch in range(1, cfg.train_epochs + 1):
        t0 = time.perf_counter()
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, x_tr.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, x_tr.shape[0] - bs + 1, bs):
            idx = perm[i : i + bs]
            key, sk = jax.random.split(key)
            c = make_clamps(x_tr[idx], y_tr[idx], [p[idx] for p in patches_tr], np_)
            graph, opt_state, loss = train_step(graph, opt_state, c, sk)
            total_loss += float(loss)
            n += 1
        dt = time.perf_counter() - t0

        if epoch % cfg.eval_every == 0 or epoch == cfg.train_epochs:
            correct, total = 0, 0
            for i in range(0, x_te.shape[0] - bs + 1, bs):
                key, ek2 = jax.random.split(key)
                c_eval = {"image": x_te[i : i + bs]}
                for pi in range(np_):
                    c_eval[f"patch_{pi}"] = patches_te[pi][i : i + bs]
                preds = eval_step(graph, c_eval, ek2)
                correct += int(
                    jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y_te[i : i + bs], -1))
                )
                total += bs
            acc = correct / total if total > 0 else 0.0
            accs.append(acc)
            print(f"    Epoch {epoch}/{cfg.train_epochs}  loss={total_loss / n:.2f}  acc={acc:.4f}  ({dt:.1f}s)")
        else:
            print(f"    Epoch {epoch}/{cfg.train_epochs}  loss={total_loss / n:.2f}  ({dt:.1f}s)")

    return accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: FashionSearchConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    print("Loading Fashion-MNIST...")
    x_tr, y_tr, x_te, y_te, patches_tr, patches_te = load_fashion_mnist()

    key, bk = jax.random.split(key)
    perm = jax.random.permutation(bk, x_tr.shape[0])
    idx = perm[: cfg.batch_size]
    x_batch = x_tr[idx]
    y_batch = y_tr[idx]
    patches_batch = [p[idx] for p in patches_tr]

    variables = make_variables(cfg)
    candidates = make_candidate_pool(variables, cfg)
    print(f"Variables: {[v.name for v in variables]}")
    print(f"Candidate pool: {len(candidates)} edges")
    print(f"  Transform types: {sorted(set(c.transform_type for c in candidates))}")

    # --- Residual-driven search ---
    print("\n=== Residual-Driven Search ===")
    key, rk = jax.random.split(key)
    res_graph, res_history = run_search(
        cfg, variables, candidates, x_batch, y_batch, patches_batch, rk, "residual"
    )
    res_edges = [t.id for t in res_graph.transforms]
    print(f"Final architecture ({len(res_edges)} edges): {res_edges}")

    # --- Random search ---
    rand_histories, rand_graphs = [], []
    for trial in range(cfg.n_random_trials):
        print(f"\n=== Random Search (trial {trial + 1}/{cfg.n_random_trials}) ===")
        key, rk = jax.random.split(key)
        rg, rh = run_search(
            cfg, variables, candidates, x_batch, y_batch, patches_batch, rk, "random"
        )
        rand_histories.append(rh)
        rand_graphs.append(rg)

    # --- Plots ---
    print("\n=== Generating plots ===")
    plot_search_comparison(
        res_history, rand_histories,
        os.path.join(cfg.output_dir, "search_comparison.png"),
    )
    plot_score_vs_step(
        res_history, rand_histories,
        os.path.join(cfg.output_dir, "score_vs_step.png"),
    )
    plot_energy_gap(
        res_history, rand_histories,
        os.path.join(cfg.output_dir, "energy_gap.png"),
    )
    if res_history["eigenvalues"]:
        plot_eigenspectrum(
            res_history["eigenvalues"],
            os.path.join(cfg.output_dir, "eigenspectrum.png"),
        )
    plot_transform_types(
        res_edges,
        [[t.id for t in rg.transforms] for rg in rand_graphs],
        os.path.join(cfg.output_dir, "transform_types.png"),
    )

    # --- Train and evaluate ---
    print("\n=== Training residual-driven architecture ===")
    key, tk = jax.random.split(key)
    res_accs = train_and_evaluate(
        res_graph, cfg, x_tr, y_tr, x_te, y_te, patches_tr, patches_te, tk
    )

    rand_accs_list = []
    for trial, rg in enumerate(rand_graphs):
        print(f"\n=== Training random architecture (trial {trial + 1}) ===")
        key, tk = jax.random.split(key)
        ra = train_and_evaluate(
            rg, cfg, x_tr, y_tr, x_te, y_te, patches_tr, patches_te, tk
        )
        rand_accs_list.append(ra)

    if res_accs:
        plot_training_curves(
            res_accs, rand_accs_list,
            os.path.join(cfg.output_dir, "training_curves.png"),
        )

    # --- Save ---
    results = {
        "config": vars(cfg),
        "residual_driven": {
            "history": {k: v for k, v in res_history.items() if k != "eigenvalues"},
            "edges": res_edges,
            "final_accs": res_accs,
        },
        "random": [
            {
                "history": {k: v for k, v in rh.items() if k != "eigenvalues"},
                "edges": [t.id for t in rg.transforms],
                "final_accs": ra,
            }
            for rh, rg, ra in zip(rand_histories, rand_graphs, rand_accs_list)
        ],
    }
    save_results(results, cfg.output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Residual-driven: {len(res_graph.transforms)} edges")
    print(f"  Edges: {res_edges}")
    if res_accs:
        print(f"  Best test accuracy: {max(res_accs):.4f}")
    for trial in range(len(rand_graphs)):
        edges = [t.id for t in rand_graphs[trial].transforms]
        print(f"Random trial {trial + 1}: {len(edges)} edges")
        if rand_accs_list[trial]:
            print(f"  Best test accuracy: {max(rand_accs_list[trial]):.4f}")
    print(f"\nOutputs saved to {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(FashionSearchConfig)
    main(cfg)
