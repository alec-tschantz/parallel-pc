"""Structure learning via A-optimal backward elimination.

J(S) = tr(H_S^{-1} Σ_task)

Single forward pass for Jacobians, one probe batch for task covariance,
then iteratively remove edges that least increase the score.
No inference, no training.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .engine import init
from .graph import Graph
from .metrics import (
    classify_edges,
    precompute_edge_data,
    task_covariance,
    score_edge_set,
    score_each_removal,
)


@dataclass
class SearchConfig:
    eps: float = 1e-4  # regularisation (= 1/ηT spectral gate)
    delta: float = 0.0  # stopping tolerance (0 = stop when any removal hurts)


def _build_subgraph(graph: Graph, keep_indices: list[int]) -> Graph:
    kept_t = [graph.transforms[i] for i in keep_indices]
    kept_tids = {graph.transforms[i].id for i in keep_indices}
    kept_e = [
        e
        for e in graph.energies
        if all(
            a in kept_tids or not any(a == t.id for t in graph.transforms)
            for a in e.args
        )
    ]
    return Graph(variables=list(graph.variables), transforms=kept_t, energies=kept_e)


def reduce(
    graph: Graph,
    clamps: dict[str, jax.Array],
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[Graph, dict]:
    """Backward elimination via A-optimal score.

    Single forward pass → Jacobians → task covariance → iteratively remove
    internal edges that least increase the score.
    Stops when every removal would increase score by more than δ.

    Returns (reduced_graph, diagnostics).
    """
    state = init(graph, clamps, key=key)
    all_J, all_r = precompute_edge_data(graph, state)
    edges = classify_edges(graph, state)
    boundary_idx = edges["boundary"]
    internal_idx = edges["internal"]

    Sigma_task = task_covariance(all_J, all_r, boundary_idx)

    D = all_J[next(iter(all_J))].shape[1]
    print(
        f"  {len(boundary_idx)} boundary, {len(internal_idx)} internal edges  "
        f"(D={D})"
    )

    active = list(range(len(graph.transforms)))
    internal = list(internal_idx)
    result = score_edge_set(all_J, active, Sigma_task, cfg.eps)
    current_score = result["score"]

    history = [{"n_edges": len(active), "score": current_score, "removed": None}]
    pruned_order = []
    print(f"  Full graph: score={current_score:.6f}")

    # Backward elimination: remove edge whose removal least increases score
    while internal:
        removal_scores = score_each_removal(
            all_J, active, internal, Sigma_task, cfg.eps
        )

        # Find edge whose removal gives lowest score (least damage)
        best_e = min(internal, key=lambda e: removal_scores[e])
        best_score = removal_scores[best_e]
        delta = best_score - current_score  # ≥ 0 (removing PSD term can only hurt)

        if delta > cfg.delta + 1e-10:
            print(
                f"  Stop: best removal Δ={delta:+.6f} > δ={cfg.delta}  "
                f"({len(internal)} internal remain)"
            )
            break

        name = graph.transforms[best_e].id
        active.remove(best_e)
        internal.remove(best_e)
        pruned_order.append(best_e)
        current_score = best_score

        history.append(
            {
                "n_edges": len(active),
                "score": current_score,
                "removed": best_e,
                "removed_name": name,
                "delta": delta,
            }
        )
        print(
            f"  Remove {name}: score={current_score:.6f} (Δ={delta:+.6f}), "
            f"{len(internal)} internal left"
        )

    return _build_subgraph(graph, active), {
        "history": history,
        "n_boundary": len(boundary_idx),
        "n_internal_start": len(internal_idx),
        "n_internal_final": len(internal),
        "pruned_order": pruned_order,
        "final_edges": [graph.transforms[i].id for i in active],
    }
