"""A-optimal scoring for structure learning.

J(S) = tr(H_S^{-1} Σ_task)

H_S = Σ_{e∈S} J_e^T J_e + εI is the regularised Gauss-Newton Hessian.
Σ_task is the task covariance from data-facing edge residuals projected
into state space.  Lower score = better structure.

No inference, no training — single forward pass for Jacobians.
"""

from typing import Any

import jax
import jax.numpy as jnp

from .graph import Graph
from .types import State


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
        all_J[e]: (d_tgt, D) batch-averaged Jacobian
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


def task_covariance(
    all_J: dict[int, jax.Array],
    all_r: dict[int, jax.Array],
    boundary_idx: list[int],
) -> jax.Array:
    """Task covariance from data-facing edge residuals.

    Σ_task = (1/N) Σ_n g_n g_n^T
    where g_n = Σ_{e∈boundary} J_e^T r_{e,n}

    Projects data variance into state space.
    """
    D = all_J[next(iter(all_J))].shape[1]
    N = all_r[boundary_idx[0]].shape[0]
    g = jnp.zeros((N, D))
    for e in boundary_idx:
        g = g + all_r[e] @ all_J[e]  # (N, D)
    return (g.T @ g) / N  # (D, D)


def score_edge_set(
    all_J: dict[int, jax.Array],
    edge_set: list[int],
    Sigma_task: jax.Array,
    eps: float = 1e-4,
) -> dict[str, Any]:
    """A-optimal score: J(S) = tr(H_S^{-1} Σ_task).

    H_S = Σ_{e∈S} J_e^T J_e + εI.  Lower score = better structure.

    Returns dict with score and eigenvalues of H_S.
    """
    D = Sigma_task.shape[0]
    H = eps * jnp.eye(D)
    for e in edge_set:
        H = H + all_J[e].T @ all_J[e]

    score = float(jnp.trace(jnp.linalg.solve(H, Sigma_task)))
    eigenvalues = jnp.linalg.eigvalsh(H)

    return {"score": score, "eigenvalues": eigenvalues}


def score_each_removal(
    all_J: dict[int, jax.Array],
    current_edges: list[int],
    candidates: list[int],
    Sigma_task: jax.Array,
    eps: float = 1e-4,
) -> dict[int, float]:
    """Score of S \\ {e} for each candidate e. Returns {edge_idx: score_without}."""
    scores = {}
    for e in candidates:
        reduced = [i for i in current_edges if i != e]
        scores[e] = score_edge_set(all_J, reduced, Sigma_task, eps)["score"]
    return scores
