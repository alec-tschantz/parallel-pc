"""Structure learning via boundary Schur complement backward elimination.

J(S) = tr(Γ_B*(S)^{-1} Σ_task)

No inference simulation. Single forward pass for Jacobians.
"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .engine import init
from .graph import Graph, expand
from .metrics import (
    classify_edges,
    partition_dims,
    precompute_edge_data,
    score_edge_set,
    score_each_removal,
)
from .types import Energy, Transform


@dataclass
class SearchConfig:
    eps: float = 1e-4  # ridge regularisation (= 1/ηT spectral gate)
    delta: float = 0.0  # stopping tolerance (0 = stop when any removal hurts)


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
    return t, candidate.energy_factory(tid)


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


def _build_subgraph(graph: Graph, keep_indices: list[int]) -> Graph:
    kept_t = [graph.transforms[i] for i in keep_indices]
    kept_tids = {graph.transforms[i].id for i in keep_indices}
    kept_e = [
        e for e in graph.energies
        if all(a in kept_tids or not any(a == t.id for t in graph.transforms) for a in e.args)
    ]
    return Graph(variables=list(graph.variables), transforms=kept_t, energies=kept_e)


def reduce(
    graph: Graph,
    clamps: dict[str, jax.Array],
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[Graph, dict]:
    """Backward elimination via boundary Schur complement.

    Single forward pass → Jacobians → iteratively remove internal edges
    that least increase the score. Stops when every removal exceeds tolerance.

    Returns (reduced_graph, diagnostics).
    """
    # Linearisation point: initial state (no inference)
    state = init(graph, clamps, key=key)

    # Precompute all Jacobians and residuals (single forward pass)
    all_J, all_r = precompute_edge_data(graph, state)
    edges = classify_edges(graph, state)
    boundary_idx = edges["boundary"]
    internal_idx = edges["internal"]

    B_dims, I_dims = partition_dims(graph, state, boundary_idx)
    D_B, D_I = len(B_dims), len(I_dims)
    print(f"  {len(boundary_idx)} boundary, {len(internal_idx)} internal edges  "
          f"(D_B={D_B}, D_I={D_I})")

    # Score full graph
    active = list(range(len(graph.transforms)))
    internal = list(internal_idx)
    result = score_edge_set(all_J, all_r, boundary_idx, active, B_dims, I_dims, cfg.eps)
    current_score = result["score"]

    history = [{"n_edges": len(active), "score": current_score, "removed": None}]
    pruned_order = []
    print(f"  Full graph: score={current_score:.6f}")

    # Backward elimination
    while internal:
        removal_scores = score_each_removal(
            all_J, all_r, boundary_idx, active, internal, B_dims, I_dims, cfg.eps
        )

        best_e = min(internal, key=lambda e: removal_scores[e])
        best_score = removal_scores[best_e]
        delta = best_score - current_score

        if delta > cfg.delta + 1e-10:
            print(f"  Stop: best removal Δ={delta:+.6f} > δ={cfg.delta}  "
                  f"({len(internal)} internal remain)")
            break

        name = graph.transforms[best_e].id
        active.remove(best_e)
        internal.remove(best_e)
        pruned_order.append(best_e)
        current_score = best_score

        history.append({
            "n_edges": len(active),
            "score": current_score,
            "removed": best_e,
            "removed_name": name,
            "delta": delta,
        })
        print(f"  Remove {name}: score={current_score:.6f} (Δ={delta:+.6f}), "
              f"{len(internal)} internal left")

    return _build_subgraph(graph, active), {
        "history": history,
        "n_boundary": len(boundary_idx),
        "n_internal_start": len(internal_idx),
        "n_internal_final": len(internal),
        "pruned_order": pruned_order,
        "final_edges": [graph.transforms[i].id for i in active],
    }


def random_reduce(
    graph: Graph, state, n_keep_internal: int, key: jax.Array,
) -> Graph:
    """Randomly keep n_keep_internal internal edges. All boundary edges kept."""
    edges = classify_edges(graph, state)
    internal = edges["internal"]
    if n_keep_internal >= len(internal):
        return graph
    perm = jax.random.permutation(key, len(internal))
    keep = sorted([internal[int(perm[i])] for i in range(n_keep_internal)])
    return _build_subgraph(graph, sorted(edges["boundary"] + keep))
