"""Boundary Schur complement scoring for structure learning.

J(S) = tr(Γ_B*(S)^{-1} Σ_task)

Γ_B* is the Schur complement of the precision matrix on B-type variable
dimensions after marginalising I-type dimensions. No driving force, no inference.
"""

from typing import Any

import jax
import jax.numpy as jnp

from .graph import Graph
from .types import State


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------


def classify_edges(graph: Graph, state: State) -> dict[str, list[int]]:
    """Classify edges as boundary (clamped endpoint) or internal (all free).

    Returns {"boundary": [indices], "internal": [indices]}.
    """
    free_mask = state.free_mask
    layout = graph.layout
    boundary, internal = [], []
    for i, t in enumerate(graph.transforms):
        is_boundary = False
        for name in list(t.src) + list(t.tgt):
            if float(free_mask[layout.offsets[name]]) == 0.0:
                is_boundary = True
                break
        (boundary if is_boundary else internal).append(i)
    return {"boundary": boundary, "internal": internal}


# ---------------------------------------------------------------------------
# Variable partition: B-type vs I-type
# ---------------------------------------------------------------------------


def partition_dims(
    graph: Graph,
    state: State,
    boundary_idx: list[int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Partition free variable dimensions into B-type and I-type.

    A free variable is B-type if any boundary edge touches it (src or tgt).
    A free variable is I-type otherwise.

    Returns (B_dims, I_dims) as sorted index arrays into the flat state buffer.
    """
    layout = graph.layout
    free_mask = state.free_mask

    b_var_names: set[str] = set()
    for i in boundary_idx:
        t = graph.transforms[i]
        for name in list(t.src) + list(t.tgt):
            b_var_names.add(name)

    B_dims, I_dims = [], []
    for v in graph.variables:
        o, s = layout.offsets[v.name], layout.sizes[v.name]
        if float(free_mask[o]) == 0.0:
            continue  # clamped
        dims = list(range(o, o + s))
        if v.name in b_var_names:
            B_dims.extend(dims)
        else:
            I_dims.extend(dims)

    return jnp.array(sorted(B_dims), dtype=int), jnp.array(sorted(I_dims), dtype=int)


# ---------------------------------------------------------------------------
# Per-edge data
# ---------------------------------------------------------------------------


def edge_jacobian(graph: Graph, state: State, transform_idx: int) -> jax.Array:
    """Jacobian of residual (tgt - pred) for edge w.r.t. flat state. (B, d_tgt, D)."""
    t = graph.transforms[transform_idx]
    layout = graph.layout

    def residual(flat_single):
        srcs = []
        for s in t.src:
            o, sz, sh = layout.offsets[s], layout.sizes[s], layout.shapes[s]
            srcs.append(flat_single[o : o + sz].reshape(sh))
        out = t.module(*srcs)  # type: ignore
        pred = (
            jnp.concatenate([v.ravel() for v in out])
            if isinstance(out, tuple)
            else out.ravel()
        )
        tgt = jnp.concatenate(
            [
                flat_single[layout.offsets[n] : layout.offsets[n] + layout.sizes[n]]
                for n in t.tgt
            ]
        )
        return tgt - pred

    J = jax.vmap(jax.jacrev(residual))(state.flat)
    return J * state.free_mask[None, None, :]


def precompute_edge_data(
    graph: Graph,
    state: State,
) -> tuple[dict[int, jax.Array], dict[int, jax.Array]]:
    """Batch-averaged Jacobians and per-sample residuals for all edges.

    Returns (all_J, all_r):
        all_J[e]: (d_tgt, D_full) batch-averaged Jacobian
        all_r[e]: (N, d_tgt) per-sample residuals
    """
    layout = graph.layout
    flat = state.flat
    all_J, all_r = {}, {}
    for i, t in enumerate(graph.transforms):
        all_J[i] = jnp.mean(edge_jacobian(graph, state, i), axis=0)

        def _fwd(flat_single, _t=t):
            srcs = []
            for s in _t.src:
                o, sz, sh = layout.offsets[s], layout.sizes[s], layout.shapes[s]
                srcs.append(flat_single[o : o + sz].reshape(sh))
            out = _t.module(*srcs)  # type: ignore
            return (
                jnp.concatenate([v.ravel() for v in out])
                if isinstance(out, tuple)
                else out.ravel()
            )

        pred = jax.vmap(_fwd)(flat)
        tgt = jnp.concatenate(
            [
                flat[:, layout.offsets[n] : layout.offsets[n] + layout.sizes[n]]
                for n in t.tgt
            ],
            axis=1,
        )
        all_r[i] = tgt - pred
    return all_J, all_r


# ---------------------------------------------------------------------------
# Boundary Schur complement score
# ---------------------------------------------------------------------------


def score_edge_set(
    all_J: dict[int, jax.Array],
    all_r: dict[int, jax.Array],
    boundary_idx: list[int],
    edge_set: list[int],
    B_dims: jnp.ndarray,
    I_dims: jnp.ndarray,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """Boundary Schur complement score: J(S) = tr(Γ_B*^{-1} Σ_task).

    Γ_B* is the Schur complement on B-dims after marginalising I-dims.
    Σ_task is the task covariance in B-dim state space, computed from
    boundary edge driving forces.

    Returns dict with score, eigenvalues of Γ_B*.
    """
    D_full = all_J[next(iter(all_J))].shape[1]
    D_B = len(B_dims)
    D_I = len(I_dims)

    # Build Γ_S in full space
    Gamma = eps * jnp.eye(D_full)
    for e in edge_set:
        Gamma = Gamma + all_J[e].T @ all_J[e]

    # Extract blocks
    G_BB = Gamma[jnp.ix_(B_dims, B_dims)]

    if D_I == 0:
        Gamma_B_star = G_BB
    else:
        G_BI = Gamma[jnp.ix_(B_dims, I_dims)]
        G_II = Gamma[jnp.ix_(I_dims, I_dims)]
        Gamma_B_star = G_BB - G_BI @ jnp.linalg.solve(G_II, G_BI.T)

    # Task covariance in B-dim state space:
    # g_{B,n} = Σ_{e∈boundary} (J_e^T r_{e,n})[B_dims]
    N = next(iter(all_r.values())).shape[0]
    g_B = jnp.zeros((N, D_B))
    for e in boundary_idx:
        force = all_r[e] @ all_J[e]  # (N, D_full)
        g_B = g_B + force[:, B_dims]

    Sigma_task = (g_B.T @ g_B) / N  # (D_B, D_B)

    # Score: tr(Γ_B*^{-1} Σ_task)
    score = float(jnp.trace(jnp.linalg.solve(Gamma_B_star, Sigma_task)))

    eigenvalues = jnp.linalg.eigvalsh(Gamma_B_star)

    return {"score": score, "eigenvalues": eigenvalues}


def score_each_removal(
    all_J: dict[int, jax.Array],
    all_r: dict[int, jax.Array],
    boundary_idx: list[int],
    current_edges: list[int],
    candidates: list[int],
    B_dims: jnp.ndarray,
    I_dims: jnp.ndarray,
    eps: float = 1e-4,
) -> dict[int, float]:
    """Score of S \\ {e} for each candidate e. Returns {edge_idx: score_without}."""
    scores = {}
    for e in candidates:
        reduced = [i for i in current_edges if i != e]
        scores[e] = score_edge_set(
            all_J, all_r, boundary_idx, reduced, B_dims, I_dims, eps
        )["score"]
    return scores
