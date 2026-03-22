"""Inference-aware structure reduction via leverage-score pruning.

Build a supergraph with all candidate edges.  Compute the regularised
precision-matrix inverse once.  Score every edge by its leverage — how much
of the graph's inference capacity depends uniquely on that edge.  Prune
lowest-leverage edges iteratively, stopping when the energy gap (discrimination
between correct and incorrect labels) degrades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .engine import init, infer
from .graph import Graph, expand
from .metrics import (
    decompose,
    edge_jacobian,
    leverage_scores,
    precision_inverse,
    woodbury_downdate,
)
from .types import Energy, Transform


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    eta: float = 0.05
    T: int = 20
    infer_iters: int = 30
    infer_lr: float = 0.05
    eps: float = 1e-4
    prune_fraction: float = 0.2
    score_w: float = 1.0       # weight on energy gap in Score (eq. 6)
    score_tolerance: float = 0.05  # max Score decrease per round


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    transform_factory: Callable[[jax.Array], Transform]
    energy_factory: Callable[[str], Energy]


def instantiate_candidate(
    candidate: Candidate, edge_idx: int, key: jax.Array
) -> tuple[Transform, Energy]:
    t = candidate.transform_factory(key)
    tid = f"t_e{edge_idx}_{candidate.name}"
    t = Transform(tid, t.module, src=t.src, tgt=t.tgt)
    e = candidate.energy_factory(tid)
    return t, e


def build_supergraph(
    graph: Graph, candidates: list[Candidate], key: jax.Array
) -> Graph:
    keys = jax.random.split(key, len(candidates))
    new_t, new_e = [], []
    for i, cand in enumerate(candidates):
        t, e = instantiate_candidate(cand, i, keys[i])
        new_t.append(t)
        new_e.append(e)
    return expand(graph, new_transforms=new_t, new_energies=new_e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_subgraph(graph: Graph, active_indices: list[int]) -> Graph:
    """Build a Graph keeping only edges at the given indices."""
    kept_t = [graph.transforms[i] for i in active_indices]
    kept_tids = {graph.transforms[i].id for i in active_indices}
    kept_e = [
        e for e in graph.energies
        if all(
            a in kept_tids or not any(a == t.id for t in graph.transforms)
            for a in e.args
        )
    ]
    return Graph(variables=list(graph.variables), transforms=kept_t, energies=kept_e)


def _compute_score(graph, clamps_true, clamps_wrong, cfg, key):
    """Compute Score(G) = -φ_T(true) + w·ΔE (eq. 6)."""
    optimizer = optax.adam(cfg.infer_lr)
    k1, k2 = jax.random.split(key)

    st = init(graph, clamps_true, key=k1)
    st = infer(graph, st, optimizer=optimizer, iters=cfg.infer_iters)
    dt = decompose(graph, st, cfg.eta, cfg.T)
    phi_true = float(jnp.mean(dt["phi_T_predicted"]))

    sw = init(graph, clamps_wrong, key=k2)
    sw = infer(graph, sw, optimizer=optimizer, iters=cfg.infer_iters)
    dw = decompose(graph, sw, cfg.eta, cfg.T)
    phi_wrong = float(jnp.mean(dw["phi_T_predicted"]))

    delta_e = phi_wrong - phi_true
    score = -phi_true + cfg.score_w * delta_e
    return score, phi_true, phi_wrong, delta_e


# ---------------------------------------------------------------------------
# Core: leverage-based selection
# ---------------------------------------------------------------------------


def compute_leverages(
    graph: Graph,
    clamps: dict[str, jax.Array],
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[list[float], jax.Array]:
    """Compute leverage scores for all edges. Returns (scores, M_inv)."""
    optimizer = optax.adam(cfg.infer_lr)
    key, ik = jax.random.split(key)
    state = init(graph, clamps, key=ik)
    state = infer(graph, state, optimizer=optimizer, iters=cfg.infer_iters)
    M_inv = precision_inverse(graph, state, eps=cfg.eps)
    scores = leverage_scores(graph, state, M_inv)
    return scores, M_inv


def select_top_k(
    graph: Graph,
    leverages: list[float],
    k: int,
) -> list[int]:
    """Return indices of top-k edges by leverage."""
    ranked = sorted(range(len(leverages)), key=lambda i: leverages[i], reverse=True)
    return sorted(ranked[:k])


# ---------------------------------------------------------------------------
# Reduce (Algorithm 1 with energy-gap stopping)
# ---------------------------------------------------------------------------


def reduce(
    graph: Graph,
    clamps_true: dict[str, jax.Array],
    clamps_wrong: dict[str, jax.Array],
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[Graph, dict]:
    """Iterative leverage-score reduction with energy-gap stopping.

    Uses leverage scores for ranking (which edges to prune) and
    Score = -φ_T(true) + w·ΔE for stopping (when discrimination degrades).
    """
    optimizer = optax.adam(cfg.infer_lr)

    # Initial leverage computation
    key, lk = jax.random.split(key)
    lev, M_inv = compute_leverages(graph, clamps_true, cfg, lk)

    # Initial score
    key, sk = jax.random.split(key)
    score_full, phi_t, phi_w, delta_e = _compute_score(
        graph, clamps_true, clamps_wrong, cfg, sk)

    print(f"  Full: {len(graph.transforms)} edges, Score={score_full:.4f}, "
          f"φ_T(true)={phi_t:.4f}, φ_T(wrong)={phi_w:.4f}, ΔE={delta_e:.4f}")

    active = list(range(len(graph.transforms)))
    history = [{"n_edges": len(active), "score": score_full,
                "phi_true": phi_t, "phi_wrong": phi_w, "delta_e": delta_e}]

    # Precompute Jacobians for Woodbury downdates
    key, jk = jax.random.split(key)
    state_for_jac = init(graph, clamps_true, key=jk)
    state_for_jac = infer(graph, state_for_jac, optimizer=optimizer, iters=cfg.infer_iters)
    jacobians = {}
    for i in range(len(graph.transforms)):
        J = edge_jacobian(graph, state_for_jac, i)
        jacobians[i] = jnp.mean(J, axis=0)

    prev_score = score_full
    round_num = 0

    while len(active) > 1:
        # Rank active edges by leverage
        active_lev = [(i, lev[i]) for i in active]
        active_lev.sort(key=lambda x: x[1])

        n_prune = max(1, int(len(active) * cfg.prune_fraction))
        to_remove = [i for i, _ in active_lev[:n_prune]]

        # Woodbury downdates
        for ei in to_remove:
            M_inv = woodbury_downdate(M_inv, jacobians[ei])
            active.remove(ei)

        # Recompute leverages for remaining
        for i in active:
            A = jacobians[i] @ M_inv
            lev[i] = float(jnp.sum(A * jacobians[i]))

        # Build reduced graph and evaluate Score
        reduced = _build_subgraph(graph, active)
        key, sk = jax.random.split(key)
        try:
            score_now, phi_t, phi_w, delta_e = _compute_score(
                reduced, clamps_true, clamps_wrong, cfg, sk)
        except Exception as ex:
            print(f"  Round {round_num+1}: error ({ex}), restoring")
            for i in to_remove:
                active.append(i)
            active.sort()
            break

        round_num += 1
        removed_names = [graph.transforms[i].id.split("_", 2)[-1] for i in to_remove]
        print(f"  Round {round_num}: {len(active)} edges, Score={score_now:.4f} "
              f"(Δ={score_now - prev_score:+.4f}), ΔE={delta_e:.4f}, "
              f"removed: {removed_names}")

        history.append({"n_edges": len(active), "score": score_now,
                        "phi_true": phi_t, "phi_wrong": phi_w, "delta_e": delta_e,
                        "removed": [graph.transforms[i].id for i in to_remove]})

        # Stop if Score decreased significantly
        if score_now < prev_score - cfg.score_tolerance:
            print(f"  Score dropped by {prev_score - score_now:.4f} > tolerance, restoring")
            for i in to_remove:
                active.append(i)
            active.sort()
            history.pop()
            break

        prev_score = score_now

    final = _build_subgraph(graph, active)
    return final, {
        "history": history,
        "final_leverage": {graph.transforms[i].id: lev[i] for i in active},
        "n_full": len(graph.transforms),
        "n_reduced": len(active),
    }


# ---------------------------------------------------------------------------
# Random pruning baseline
# ---------------------------------------------------------------------------


def random_reduce(
    graph: Graph, n_keep: int, key: jax.Array
) -> Graph:
    """Randomly keep n_keep edges."""
    n = len(graph.transforms)
    if n_keep >= n:
        return graph
    perm = jax.random.permutation(key, n)
    keep = sorted([int(perm[i]) for i in range(n_keep)])
    return _build_subgraph(graph, keep)
