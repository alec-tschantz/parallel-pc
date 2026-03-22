"""Inference-aware structure reduction (Algorithm 1 from the paper).

Start from a supergraph with all candidate edges.  Compute the regularised
precision-matrix inverse once.  Score every edge by its leverage — how much
of the graph's inference capacity depends uniquely on that edge.  Iteratively
prune the lowest-leverage fraction and update the inverse via Woodbury
downdates.  Stop when the profiled energy exceeds a tolerance above the full
graph's score.
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
    eps: float = 1e-4          # ridge parameter for M
    prune_fraction: float = 0.2  # ρ — fraction to remove per round
    score_tolerance: float = 0.1  # δ — max φ_T increase over full


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A potential edge to add to the supergraph."""
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
    graph: Graph,
    candidates: list[Candidate],
    key: jax.Array,
) -> Graph:
    """Add all candidates to the graph. Returns the supergraph."""
    keys = jax.random.split(key, len(candidates))
    new_transforms, new_energies = [], []
    for i, cand in enumerate(candidates):
        t, e = instantiate_candidate(cand, i, keys[i])
        new_transforms.append(t)
        new_energies.append(e)
    return expand(graph, new_transforms=new_transforms, new_energies=new_energies)


# ---------------------------------------------------------------------------
# Reduce (Algorithm 1)
# ---------------------------------------------------------------------------


def reduce(
    graph: Graph,
    clamps: dict[str, jax.Array],
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[Graph, dict]:
    """Inference-aware structure reduction.

    1. Run inference on full graph
    2. Compute M⁻¹ once
    3. Compute leverage scores for all edges
    4. Iteratively prune lowest-leverage edges (Woodbury downdates)
    5. Stop when φ_T exceeds tolerance above full graph's φ_T

    Returns (reduced_graph, diagnostics).
    """
    optimizer = optax.adam(cfg.infer_lr)

    # Step 1: inference on full graph
    key, ik = jax.random.split(key)
    state = init(graph, clamps, key=ik)
    state = infer(graph, state, optimizer=optimizer, iters=cfg.infer_iters)

    # Step 2: compute M⁻¹
    M_inv = precision_inverse(graph, state, eps=cfg.eps)

    # Step 3: initial leverage scores
    scores = leverage_scores(graph, state, M_inv)

    # Step 4: evaluate full graph's φ_T
    d = decompose(graph, state, cfg.eta, cfg.T)
    phi_full = float(jnp.mean(d["phi_T_predicted"]))
    batch_size = state.flat.shape[0]

    print(f"  Full graph: {len(graph.transforms)} edges, φ_T={phi_full:.4f}, "
          f"eff_rank={d['effective_rank']}")

    # Track which edges are still active (by index into graph.transforms)
    active = list(range(len(graph.transforms)))
    history = [{
        "n_edges": len(active),
        "phi_T": phi_full,
        "leverage_scores": {graph.transforms[i].id: scores[i] for i in active},
    }]

    # Precompute batch-averaged Jacobians (needed for Woodbury downdates)
    jacobians = {}
    for i in range(len(graph.transforms)):
        J = edge_jacobian(graph, state, i)
        jacobians[i] = jnp.mean(J, axis=0)  # (d_tgt, D)

    # Step 5: iterative pruning
    round_num = 0
    while True:
        if len(active) <= 1:
            break

        # Sort active edges by leverage
        active_scores = [(i, scores[i]) for i in active]
        active_scores.sort(key=lambda x: x[1])

        # How many to prune this round
        n_prune = max(1, int(len(active) * cfg.prune_fraction))
        to_remove = [i for i, _ in active_scores[:n_prune]]

        # Woodbury downdates
        for edge_idx in to_remove:
            J_e = jacobians[edge_idx]
            M_inv = woodbury_downdate(M_inv, J_e)
            active.remove(edge_idx)

        # Recompute leverage scores for remaining edges
        scores_new = {}
        for i in active:
            J_avg = jacobians[i]
            A = J_avg @ M_inv
            scores_new[i] = float(jnp.sum(A * J_avg))
        for i in active:
            scores[i] = scores_new[i]

        # Evaluate φ_T on reduced graph
        remaining_transforms = [graph.transforms[i] for i in active]
        remaining_energies = []
        active_tids = {graph.transforms[i].id for i in active}
        for e in graph.energies:
            if all(a in active_tids or not any(a == t.id for t in graph.transforms) for a in e.args):
                remaining_energies.append(e)

        try:
            reduced = Graph(
                variables=list(graph.variables),
                transforms=remaining_transforms,
                energies=remaining_energies,
            )
            key, rk = jax.random.split(key)
            rs = init(reduced, clamps, key=rk)
            rs = infer(reduced, rs, optimizer=optimizer, iters=cfg.infer_iters)
            rd = decompose(reduced, rs, cfg.eta, cfg.T)
            phi_reduced = float(jnp.mean(rd["phi_T_predicted"]))
        except Exception as ex:
            print(f"  Round {round_num}: error evaluating reduced graph: {ex}")
            # Restore removed edges
            for i in to_remove:
                active.append(i)
            active.sort()
            break

        removed_names = [graph.transforms[i].id for i in to_remove]
        round_num += 1
        print(f"  Round {round_num}: {len(active)} edges, φ_T={phi_reduced:.4f} "
              f"(Δ={phi_reduced - phi_full:+.4f}), "
              f"removed: {[n.split('_', 2)[-1] for n in removed_names]}")

        h = {
            "n_edges": len(active),
            "phi_T": phi_reduced,
            "delta_phi": phi_reduced - phi_full,
            "removed": removed_names,
        }
        history.append(h)

        # Check tolerance
        if phi_reduced - phi_full > cfg.score_tolerance:
            print(f"  Score tolerance exceeded (δ={phi_reduced - phi_full:.4f} > {cfg.score_tolerance}), "
                  f"restoring last batch")
            for i in to_remove:
                active.append(i)
            active.sort()
            history.pop()
            break

    # Build final reduced graph
    final_transforms = [graph.transforms[i] for i in active]
    active_tids = {graph.transforms[i].id for i in active}
    final_energies = []
    for e in graph.energies:
        if all(a in active_tids or not any(a == t.id for t in graph.transforms) for a in e.args):
            final_energies.append(e)

    reduced_graph = Graph(
        variables=list(graph.variables),
        transforms=final_transforms,
        energies=final_energies,
    )

    return reduced_graph, {
        "phi_full": phi_full,
        "history": history,
        "n_edges_full": len(graph.transforms),
        "n_edges_reduced": len(active),
        "final_leverage": {graph.transforms[i].id: scores[i] for i in active},
    }


# ---------------------------------------------------------------------------
# Random pruning baseline
# ---------------------------------------------------------------------------


def random_reduce(
    graph: Graph,
    clamps: dict[str, jax.Array],
    n_keep: int,
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[Graph, dict]:
    """Randomly keep n_keep edges from the graph."""
    n_total = len(graph.transforms)
    if n_keep >= n_total:
        return graph, {"kept": list(range(n_total))}

    key, rk = jax.random.split(key)
    perm = jax.random.permutation(rk, n_total)
    keep = sorted([int(perm[i]) for i in range(n_keep)])

    kept_transforms = [graph.transforms[i] for i in keep]
    kept_tids = {graph.transforms[i].id for i in keep}
    kept_energies = []
    for e in graph.energies:
        if all(a in kept_tids or not any(a == t.id for t in graph.transforms) for a in e.args):
            kept_energies.append(e)

    reduced = Graph(
        variables=list(graph.variables),
        transforms=kept_transforms,
        energies=kept_energies,
    )
    return reduced, {"kept": keep, "edges": [graph.transforms[i].id for i in keep]}
