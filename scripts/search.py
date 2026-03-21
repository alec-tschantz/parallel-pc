"""Residual-driven structure search (Algorithm 2) on MNIST.

Compares greedy residual-driven expansion against random search baseline.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial

os.environ.setdefault("KERAS_BACKEND", "jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from keras.datasets import mnist as mnist_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ppc
from scripts.search_utils import (
    plot_eigenspectrum,
    plot_energy_gap,
    plot_score_vs_step,
    plot_search_comparison,
    plot_training_curves,
    save_results,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    seed: int = 0
    batch_size: int = 128
    n_hidden: int = 5
    hidden_dim: int = 64
    max_steps: int = 12
    epsilon_r: float = 0.001
    eta: float = 0.05
    T: int = 20
    infer_iters: int = 20
    infer_lr: float = 0.05
    top_k: int = 10
    # Training after search
    train_epochs: int = 20
    train_lr: float = 5e-5
    train_batch_size: int = 64
    eval_every: int = 5
    # Random baseline
    n_random_trials: int = 5
    output_dir: str = "outputs/search"


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


# ---------------------------------------------------------------------------
# MNIST helpers
# ---------------------------------------------------------------------------


def load_mnist():
    (x_tr, y_tr), (x_te, y_te) = mnist_data.load_data()
    x_tr = jnp.array(x_tr.reshape(-1, 784).astype("float32") / 255.0)
    x_te = jnp.array(x_te.reshape(-1, 784).astype("float32") / 255.0)
    return x_tr, jax.nn.one_hot(y_tr, 10), x_te, jax.nn.one_hot(y_te, 10)


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def make_variables(cfg: SearchConfig) -> list[ppc.Variable]:
    variables = [ppc.Variable("image", (784,)), ppc.Variable("output", (10,))]
    for i in range(cfg.n_hidden):
        variables.append(ppc.Variable(f"h{i}", (cfg.hidden_dim,)))
    return variables


def make_candidate_pool(variables: list[ppc.Variable]) -> list[Candidate]:
    """Generate all candidate edges between existing variables."""
    var_map = {v.name: v for v in variables}
    hidden_names = [v.name for v in variables if v.name.startswith("h")]
    candidates = []

    # image -> h_j
    for h in hidden_names:
        candidates.append(Candidate(
            name=f"image->{h}",
            src="image", tgt=h,
            in_dim=var_map["image"].shape[0],
            out_dim=var_map[h].shape[0],
        ))

    # h_i -> h_j (i != j)
    for h_src in hidden_names:
        for h_tgt in hidden_names:
            if h_src != h_tgt:
                candidates.append(Candidate(
                    name=f"{h_src}->{h_tgt}",
                    src=h_src, tgt=h_tgt,
                    in_dim=var_map[h_src].shape[0],
                    out_dim=var_map[h_tgt].shape[0],
                ))

    # h_i -> output
    for h in hidden_names:
        candidates.append(Candidate(
            name=f"{h}->output",
            src=h, tgt="output",
            in_dim=var_map[h].shape[0],
            out_dim=var_map["output"].shape[0],
        ))

    # image -> output (direct)
    candidates.append(Candidate(
        name="image->output",
        src="image", tgt="output",
        in_dim=var_map["image"].shape[0],
        out_dim=var_map["output"].shape[0],
    ))

    return candidates


def instantiate_candidate(
    candidate: Candidate, edge_idx: int, key: jax.Array
) -> tuple[ppc.Transform, ppc.Energy]:
    """Create a Transform and Energy for a candidate edge."""
    module = eqx.nn.Linear(candidate.in_dim, candidate.out_dim, key=key)
    tid = f"t_e{edge_idx}_{candidate.name}"
    transform = ppc.Transform(tid, module, src=candidate.src, tgt=candidate.tgt)
    energy = ppc.Energy(ppc.mse_energy, args=[tid, candidate.tgt])
    return transform, energy


def make_initial_graph(
    variables: list[ppc.Variable], key: jax.Array
) -> tuple[ppc.Graph, int]:
    """Create initial graph with a minimal path: image -> h0 -> output."""
    k1, k2 = jax.random.split(key)
    h0_dim = variables[2].shape[0]  # h0
    t0 = ppc.Transform(
        "t_e0_image->h0",
        eqx.nn.Linear(784, h0_dim, key=k1),
        src="image", tgt="h0",
    )
    t1 = ppc.Transform(
        "t_e1_h0->output",
        eqx.nn.Linear(h0_dim, 10, key=k2),
        src="h0", tgt="output",
    )
    e0 = ppc.Energy(ppc.mse_energy, args=["t_e0_image->h0", "h0"])
    e1 = ppc.Energy(ppc.mse_energy, args=["t_e1_h0->output", "output"])
    graph = ppc.Graph(
        variables=variables,
        transforms=[t0, t1],
        energies=[e0, e1],
    )
    return graph, 2  # next edge index


# ---------------------------------------------------------------------------
# Search core
# ---------------------------------------------------------------------------


def score_candidates(
    graph: ppc.Graph,
    decomp: dict,
    candidates: list[Candidate],
    clamps: dict[str, jax.Array],
    edge_idx: int,
    cfg: SearchConfig,
    key: jax.Array,
) -> list[tuple[int, float]]:
    """Score ALL candidates by energy gap (correct vs wrong labels).

    The energy gap measures discriminative power: a good edge makes the
    architecture produce lower energy for correct labels than wrong labels.
    Returns list of (candidate_idx, gain) sorted descending.
    """
    infer_opt = optax.adam(cfg.infer_lr)
    batch_size = clamps["image"].shape[0]
    keys = jax.random.split(key, len(candidates) + 1)
    perm_key = keys[-1]
    keys = keys[:-1]

    # Generate wrong labels by shuffling
    wrong_perm = jax.random.permutation(perm_key, batch_size)
    clamps_wrong = {**clamps, "output": clamps["output"][wrong_perm]}

    # Score all candidates by energy gap
    gains = []
    for i, cand in enumerate(candidates):
        k1, k2, k3 = jax.random.split(keys[i], 3)
        t, e = instantiate_candidate(cand, edge_idx + i, k1)
        expanded = ppc.expand(graph, new_transforms=[t], new_energies=[e])
        try:
            # Energy with correct labels
            s_true = ppc.init(expanded, clamps, key=k2)
            s_true = ppc.infer(expanded, s_true, optimizer=infer_opt, iters=cfg.infer_iters)
            e_true = float(ppc.energy(expanded, s_true)) / batch_size

            # Energy with wrong labels
            s_wrong = ppc.init(expanded, clamps_wrong, key=k3)
            s_wrong = ppc.infer(expanded, s_wrong, optimizer=infer_opt, iters=cfg.infer_iters)
            e_wrong = float(ppc.energy(expanded, s_wrong)) / batch_size

            # Energy gap: higher = more discriminative
            gain = e_wrong - e_true
        except Exception:
            gain = -1e9
        gains.append((i, gain))

    gains.sort(key=lambda x: x[1], reverse=True)
    return gains


def run_search(
    cfg: SearchConfig,
    variables: list[ppc.Variable],
    candidates: list[Candidate],
    x_batch: jax.Array,
    y_batch: jax.Array,
    key: jax.Array,
    mode: str = "residual",
) -> tuple[ppc.Graph, dict]:
    """Run structure search. mode='residual' or 'random'."""
    key, gk = jax.random.split(key)
    graph, edge_idx = make_initial_graph(variables, gk)
    remaining = list(range(len(candidates)))
    # Remove bootstrap edges from candidate pool
    bootstrap_names = {"image->h0", "h0->output"}
    for i, c in enumerate(candidates):
        if c.name in bootstrap_names and i in remaining:
            remaining.remove(i)

    infer_opt = optax.adam(cfg.infer_lr)
    history = {
        "coverage_gap": [],
        "conditioning_penalty": [],
        "phi_T": [],
        "effective_rank": [],
        "eigenvalues": [],
        "energy_true": [],
        "energy_wrong": [],
        "energy_gap": [],
        "selected_edge": [],
        "gain": [],
    }

    for step in range(cfg.max_steps):
        if not remaining:
            print(f"  Step {step}: No more candidates")
            break

        # Init and infer
        key, ik = jax.random.split(key)
        clamps = {"image": x_batch, "output": y_batch}
        state = ppc.init(graph, clamps, key=ik)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters)

        # Decompose
        decomp = ppc.decompose(graph, state, cfg.eta, cfg.T)
        cg = float(jnp.mean(decomp["coverage_gap"]))
        cp = float(jnp.mean(decomp["conditioning_penalty"]))
        phi = float(jnp.mean(decomp["phi_T_predicted"]))
        er = decomp["effective_rank"]
        eigs = np.array(decomp["eigenvalues"])

        # Energy gap: correct vs wrong labels
        e_true = float(ppc.energy(graph, state)) / x_batch.shape[0]
        key, wk, wik = jax.random.split(key, 3)
        wrong_perm = jax.random.permutation(wk, x_batch.shape[0])
        clamps_wrong = {"image": x_batch, "output": y_batch[wrong_perm]}
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
            print(f"  Coverage gap below threshold ({cfg.epsilon_r}), stopping")
            history["selected_edge"].append(None)
            history["gain"].append(0.0)
            break

        # Select candidate
        pool = [candidates[i] for i in remaining]
        if mode == "residual":
            key, sk = jax.random.split(key)
            scored = score_candidates(
                graph, decomp, pool, clamps, edge_idx, cfg, sk
            )
            best_pool_idx, best_gain = scored[0]
            selected_remaining_idx = best_pool_idx
        else:
            key, rk = jax.random.split(key)
            selected_remaining_idx = int(jax.random.randint(rk, (), 0, len(pool)))
            best_gain = 0.0

        selected_cand = pool[selected_remaining_idx]
        history["selected_edge"].append(selected_cand.name)
        history["gain"].append(best_gain)

        # Expand graph
        key, ek = jax.random.split(key)
        t, e = instantiate_candidate(selected_cand, edge_idx, ek)
        graph = ppc.expand(graph, new_transforms=[t], new_energies=[e])
        edge_idx += 1

        # Remove from remaining
        remaining.remove(remaining[selected_remaining_idx])

    return graph, history


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(
    graph: ppc.Graph,
    cfg: SearchConfig,
    x_tr: jax.Array,
    y_tr: jax.Array,
    x_te: jax.Array,
    y_te: jax.Array,
    key: jax.Array,
) -> list[float]:
    """Train a discovered architecture and return per-epoch test accuracy."""
    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))
    bs = cfg.train_batch_size

    @eqx.filter_jit
    def train_step(graph, opt_state, images, labels, key):
        state = ppc.init(graph, {"image": images, "output": labels}, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters)
        loss = ppc.energy(graph, state)
        grads = ppc.param_grad(graph, state)
        updates, opt_state = train_opt.update(
            eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(graph, eqx.is_array)
        )
        graph = eqx.apply_updates(graph, updates)
        return graph, opt_state, loss

    @eqx.filter_jit
    def eval_step(graph, images, key):
        state = ppc.init(graph, {"image": images}, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=cfg.infer_iters * 2)
        return ppc.variable(graph, state, "output")

    # Warmup
    key, sk = jax.random.split(key)
    graph, opt_state, _ = train_step(graph, opt_state, x_tr[:bs], y_tr[:bs], sk)

    accs = []
    for epoch in range(1, cfg.train_epochs + 1):
        t0 = time.perf_counter()
        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, x_tr.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, x_tr.shape[0] - bs + 1, bs):
            idx = perm[i : i + bs]
            key, sk = jax.random.split(key)
            graph, opt_state, loss = train_step(graph, opt_state, x_tr[idx], y_tr[idx], sk)
            total_loss += float(loss)
            n += 1
        dt = time.perf_counter() - t0

        if epoch % cfg.eval_every == 0 or epoch == cfg.train_epochs:
            correct, total = 0, 0
            for i in range(0, x_te.shape[0] - bs + 1, bs):
                key, ek2 = jax.random.split(key)
                preds = eval_step(graph, x_te[i : i + bs], ek2)
                correct += int(
                    jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y_te[i : i + bs], -1))
                )
                total += bs
            acc = correct / total if total > 0 else 0.0
            accs.append(acc)
            print(
                f"    Epoch {epoch}/{cfg.train_epochs}  loss={total_loss / n:.2f}  "
                f"acc={acc:.4f}  ({dt:.1f}s)"
            )
        else:
            print(
                f"    Epoch {epoch}/{cfg.train_epochs}  loss={total_loss / n:.2f}  ({dt:.1f}s)"
            )

    return accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: SearchConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)

    print("Loading MNIST...")
    x_tr, y_tr, x_te, y_te = load_mnist()

    # Take a batch for search
    key, bk = jax.random.split(key)
    perm = jax.random.permutation(bk, x_tr.shape[0])
    x_batch = x_tr[perm[: cfg.batch_size]]
    y_batch = y_tr[perm[: cfg.batch_size]]

    variables = make_variables(cfg)
    candidates = make_candidate_pool(variables)
    print(f"Variables: {[v.name for v in variables]}")
    print(f"Candidate pool: {len(candidates)} edges")

    # --- Residual-driven search ---
    print("\n=== Residual-Driven Search ===")
    key, rk = jax.random.split(key)
    res_graph, res_history = run_search(
        cfg, variables, candidates, x_batch, y_batch, rk, mode="residual"
    )
    print(f"Final architecture: {[t.id for t in res_graph.transforms]}")

    # --- Random search baseline ---
    rand_histories = []
    rand_graphs = []
    for trial in range(cfg.n_random_trials):
        print(f"\n=== Random Search (trial {trial + 1}/{cfg.n_random_trials}) ===")
        key, rk = jax.random.split(key)
        rg, rh = run_search(
            cfg, variables, candidates, x_batch, y_batch, rk, mode="random"
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

    # --- Train and evaluate ---
    print("\n=== Training residual-driven architecture ===")
    key, tk = jax.random.split(key)
    res_accs = train_and_evaluate(res_graph, cfg, x_tr, y_tr, x_te, y_te, tk)

    rand_accs_list = []
    for trial, rg in enumerate(rand_graphs):
        print(f"\n=== Training random architecture (trial {trial + 1}) ===")
        key, tk = jax.random.split(key)
        ra = train_and_evaluate(rg, cfg, x_tr, y_tr, x_te, y_te, tk)
        rand_accs_list.append(ra)

    if res_accs:
        plot_training_curves(
            res_accs, rand_accs_list,
            os.path.join(cfg.output_dir, "training_curves.png"),
        )

    # --- Save results ---
    results = {
        "config": {k: v for k, v in vars(cfg).items()},
        "residual_driven": {
            "history": {k: v for k, v in res_history.items() if k != "eigenvalues"},
            "edges": [t.id for t in res_graph.transforms],
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
    print(f"Residual-driven: {len(res_graph.transforms)} edges, "
          f"final coverage_gap={res_history['coverage_gap'][-1]:.4f}")
    if res_accs:
        print(f"  Best test accuracy: {max(res_accs):.4f}")
    for trial, (rh, ra) in enumerate(zip(rand_histories, rand_accs_list)):
        print(f"Random trial {trial + 1}: {len(rand_graphs[trial].transforms)} edges, "
              f"final coverage_gap={rh['coverage_gap'][-1]:.4f}")
        if ra:
            print(f"  Best test accuracy: {max(ra):.4f}")
    print(f"\nOutputs saved to {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = tyro.cli(SearchConfig)
    main(cfg)
