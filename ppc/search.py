from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .engine import init, infer
from .graph import Graph, expand
from .metrics import (
    boundary_residual,
    classify_edges,
    edge_jacobian,
    edge_precision,
    frozen_boundary_phi,
    leverage_scores,
    precision_inverse,
    woodbury_downdate,
)
from .types import Energy, Transform, State


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SearchConfig:
    eta: float = 0.05
    T: int = 20
    infer_iters: int = 10
    infer_lr: float = 0.05
    eps: float = 1e-4
    prune_fraction: float = 0.2
    score_tolerance: float = 0.15  # max fractional φ_T^B increase before stopping


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


def _build_subgraph(graph: Graph, keep_indices: list[int]) -> Graph:
    """Build Graph keeping only edges at given indices."""
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


# ---------------------------------------------------------------------------
# Reduce (Algorithm 1)
# ---------------------------------------------------------------------------


def reduce(
    graph: Graph,
    clamps: dict[str, jax.Array],
    cfg: SearchConfig,
    key: jax.Array,
) -> tuple[Graph, dict]:
    """Boundary-evaluated leverage reduction with frozen driving force.

    1. Run inference on full supergraph
    2. Classify edges → boundary (kept) + internal (prunable)
    3. Freeze: boundary residual b_B, Jacobians, driving force d = A^T b
    4. Compute M⁻¹ once
    5. Iteratively prune lowest-leverage internal edges
    6. Stop when frozen φ_T^B increases by > δ

    Returns (reduced_graph, diagnostics).
    """
    optimizer = optax.adam(cfg.infer_lr)

    # Step 1: inference on full supergraph
    key, ik = jax.random.split(key)
    state = init(graph, clamps, key=ik)
    state = infer(graph, state, optimizer=optimizer, iters=cfg.infer_iters)

    # Step 2: classify edges
    edges = classify_edges(graph, state)
    boundary_idx = edges["boundary"]
    internal_idx = edges["internal"]
    print(f"  Edges: {len(boundary_idx)} boundary, {len(internal_idx)} internal")

    # Step 3: freeze boundary residual and compute driving force
    b_B = boundary_residual(graph, state)  # (B, m_B)

    # Boundary Jacobian (batch-averaged)
    A_B_blocks = []
    for i in boundary_idx:
        J = edge_jacobian(graph, state, i)
        A_B_blocks.append(jnp.mean(J, axis=0))
    A_B = (
        jnp.concatenate(A_B_blocks, axis=0)
        if A_B_blocks
        else jnp.zeros((0, state.flat.shape[1]))
    )

    # Full weighted Jacobian for driving force
    all_J = {}
    for i in range(len(graph.transforms)):
        J = edge_jacobian(graph, state, i)
        all_J[i] = jnp.mean(J, axis=0)  # (d_tgt, D)

    # Full A (batch-averaged)
    A_full = jnp.concatenate([all_J[i] for i in range(len(graph.transforms))], axis=0)
    b_full = jnp.mean(
        jnp.concatenate(
            [edge_jacobian(graph, state, i) for i in range(len(graph.transforms))],
            axis=1,
        ),
        axis=0,
    )  # Hmm, this is A, not b. Let me compute d = A^T b properly.

    # Actually: d_frozen = Σ_e J_e^T r_e (state-space driving force)
    # This is the gradient of energy w.r.t. state, which equals A^T b
    # where b is the task residual. Let's compute it directly.
    from .engine import state_grad

    d_frozen = jnp.mean(state_grad(graph, state), axis=0)  # (D,) batch-averaged
    # Note: state_grad includes the free_mask, so d_frozen is already masked.
    # But for the frozen-RHS computation we want the unmasked gradient.
    # Actually d_frozen should be the driving force BEFORE masking.
    # d = A^T b where A is the weighted Jacobian and b is the task residual.
    # Let's compute it properly:
    from .metrics import task_residual

    b_all = jnp.mean(task_residual(graph, state), axis=0)  # (m,)
    d_frozen = A_full.T @ b_all  # (D,)

    # Step 4: M⁻¹
    M_inv = precision_inverse(graph, state, eps=cfg.eps)

    # Step 5: initial φ_T^B (exact, frozen RHS)
    Gamma_full = sum(all_J[i].T @ all_J[i] for i in range(len(graph.transforms)))
    phi_init = frozen_boundary_phi(Gamma_full, b_B, A_B, d_frozen, cfg.eta, cfg.T)  # type: ignore
    phi_prev = phi_init["phi_T_B"]

    print(
        f"  Full: {len(graph.transforms)} edges, φ_T^B={phi_prev:.4f}, "
        f"cov_gap={phi_init['coverage_gap']:.4f}, cond_pen={phi_init['conditioning_penalty']:.4f}"
    )

    # Track active internal edges
    active_internal = list(internal_idx)
    history = [
        {
            "n_internal": len(active_internal),
            "n_total": len(boundary_idx) + len(active_internal),
            **phi_init,
        }
    ]

    # Step 6: iterative pruning
    round_num = 0
    while active_internal:
        # Compute trace leverage for active internal edges
        levs = leverage_scores(graph, state, M_inv, active_internal)

        # Sort ascending
        sorted_edges = sorted(active_internal, key=lambda i: levs[i])

        # Remove bottom ρ fraction
        n_prune = max(1, int(len(active_internal) * cfg.prune_fraction))
        to_remove = sorted_edges[:n_prune]

        # Woodbury downdates
        for ei in to_remove:
            M_inv = woodbury_downdate(M_inv, all_J[ei])
            active_internal.remove(ei)

        # Compute reduced Gamma from remaining edges
        active_all = sorted(boundary_idx + active_internal)
        Gamma_reduced = (
            sum(all_J[i].T @ all_J[i] for i in active_all)
            if active_all
            else jnp.zeros_like(Gamma_full)
        )

        # Exact φ_T^B with frozen RHS
        phi_result = frozen_boundary_phi(
            Gamma_reduced, b_B, A_B, d_frozen, cfg.eta, cfg.T  # type: ignore
        )
        phi_curr = phi_result["phi_T_B"]

        round_num += 1
        removed_names = [graph.transforms[i].id.split("_", 2)[-1] for i in to_remove]
        delta = phi_curr - phi_prev
        print(
            f"  Round {round_num}: {len(active_internal)} internal, "
            f"φ_T^B={phi_curr:.4f} (Δ={delta:+.4f}), "
            f"removed: {removed_names}"
        )

        h = {
            "n_internal": len(active_internal),
            "n_total": len(boundary_idx) + len(active_internal),
            "removed": [graph.transforms[i].id for i in to_remove],
            "leverages": {graph.transforms[i].id: levs[i] for i in to_remove},
            **phi_result,
        }
        history.append(h)

        # Monotonicity check
        if delta < -1e-6:
            print(
                f"  WARNING: φ_T^B decreased by {-delta:.6f} — monotonicity violated!"
            )

        # Stopping criterion: cumulative increase exceeds fraction of initial φ_T^B
        if (phi_curr - phi_init["phi_T_B"]) / max(
            phi_init["phi_T_B"], 1e-6
        ) > cfg.score_tolerance:
            frac = (phi_curr - phi_init["phi_T_B"]) / max(phi_init["phi_T_B"], 1e-6)
            print(
                f"  Tolerance exceeded (cumulative={frac:.1%} > {cfg.score_tolerance:.0%}), restoring"
            )
            for i in to_remove:
                active_internal.append(i)
            active_internal.sort()
            history.pop()
            break

        phi_prev = phi_curr

    # Build final graph
    final_indices = sorted(boundary_idx + active_internal)
    final_graph = _build_subgraph(graph, final_indices)

    # Final leverage scores for surviving internal edges
    final_levs = (
        leverage_scores(graph, state, M_inv, active_internal) if active_internal else {}
    )

    return final_graph, {
        "history": history,
        "n_boundary": len(boundary_idx),
        "n_internal_start": len(internal_idx),
        "n_internal_final": len(active_internal),
        "boundary_edges": [graph.transforms[i].id for i in boundary_idx],
        "final_internal": [graph.transforms[i].id for i in active_internal],
        "final_leverage": {
            graph.transforms[i].id: final_levs.get(i, 0) for i in active_internal
        },
    }


# ---------------------------------------------------------------------------
# Random pruning baseline (internal edges only)
# ---------------------------------------------------------------------------


def random_reduce(
    graph: Graph, state: State, n_keep_internal: int, key: jax.Array
) -> Graph:
    """Randomly keep n_keep_internal internal edges. All boundary edges kept."""
    edges = classify_edges(graph, state)
    internal = edges["internal"]
    if n_keep_internal >= len(internal):
        return graph
    perm = jax.random.permutation(key, len(internal))
    keep_internal = sorted([internal[int(perm[i])] for i in range(n_keep_internal)])
    keep_all = sorted(edges["boundary"] + keep_internal)
    return _build_subgraph(graph, keep_all)
