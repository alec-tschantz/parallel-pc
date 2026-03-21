"""Inference precision matrix metrics: Jacobians, eigenspectra, coverage/conditioning decomposition.

Γ_G = Σ_e J_e^T Λ_e J_e   (inference precision matrix)
φ_T = ||b_⊥||² + Σ (1-ηλ_i)^{2T} c_i²   (coverage gap + conditioning penalty)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .graph import Graph
from .types import State


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


def edge_jacobian(graph: Graph, state: State, transform_idx: int) -> jax.Array:
    """Jacobian of a single transform's output w.r.t. the flat state buffer.

    Columns for clamped variables are zeroed (Jacobian is w.r.t. free states only).
    Returns shape (B, d_tgt, D).
    """
    t = graph.transforms[transform_idx]
    layout = graph.layout

    def fwd(flat_single):
        srcs = []
        for s in t.src:
            o, sz, sh = layout.offsets[s], layout.sizes[s], layout.shapes[s]
            srcs.append(flat_single[o : o + sz].reshape(sh))
        out = t.module(*srcs)  #  type: ignore
        if isinstance(out, tuple):
            return jnp.concatenate([v.ravel() for v in out])
        return out.ravel()

    J = jax.vmap(jax.jacrev(fwd))(state.flat)
    return J * state.free_mask[None, None, :]


def weighted_jacobian(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Weighted state Jacobian A_G = [Λ_1^{1/2} J_1; ...; Λ_M^{1/2} J_M].

    Shape (B, m, D) where m = Σ d_tgt(e).
    curvatures: optional {transform_id: Λ_e} (full curvature matrix, not sqrt).
    Default (None) means Λ=I (MSE energy).
    """
    blocks = []
    for i, t in enumerate(graph.transforms):
        J = edge_jacobian(graph, state, i)
        if curvatures is not None and t.id in curvatures:
            # Cholesky: Λ = L L^T, so L^T is the square-root factor: (L^T)^T (L^T) = Λ
            L = jnp.linalg.cholesky(curvatures[t.id])
            J = jnp.einsum("ji,bjk->bik", L, J)  # L^T @ J
        blocks.append(J)
    return jnp.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Precision matrix
# ---------------------------------------------------------------------------


def edge_precision(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> list[jax.Array]:
    """Per-edge PSD contributions J_e^T Λ_e J_e, each (D, D), averaged over batch."""
    terms = []
    for i, t in enumerate(graph.transforms):
        J = edge_jacobian(graph, state, i)
        if curvatures is not None and t.id in curvatures:
            gram = jnp.einsum("bji,jk,bkl->bil", J, curvatures[t.id], J)
        else:
            gram = jnp.einsum("bji,bjk->bik", J, J)
        terms.append(jnp.mean(gram, axis=0))
    return terms


def precision_matrix(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Inference precision matrix Γ = Σ J_e^T Λ_e J_e. Shape (D, D), batch-averaged."""
    return sum(edge_precision(graph, state, curvatures))  # type: ignore


def eigendecompose(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Eigenvalues and eigenvectors of precision matrix, sorted descending.

    Returns (eigenvalues (D,), eigenvectors (D, D)) where columns are eigenvectors.
    """
    G = precision_matrix(graph, state, curvatures)
    vals, vecs = jnp.linalg.eigh(G)
    return vals[::-1], vecs[:, ::-1]


# ---------------------------------------------------------------------------
# Task residual and decomposition
# ---------------------------------------------------------------------------


def task_residual(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Stacked curvature-weighted prediction errors: Λ_e^{1/2}(f_e(x_src) - x_tgt).

    Shape (B, m) where m = Σ d_tgt(e).
    """
    layout = graph.layout
    flat = state.flat
    blocks = []
    for t in graph.transforms:

        def _fwd(flat_single, _t=t):
            srcs = []
            for s in _t.src:
                o, sz, sh = layout.offsets[s], layout.sizes[s], layout.shapes[s]
                srcs.append(flat_single[o : o + sz].reshape(sh))
            out = _t.module(*srcs)  # type: ignore
            if isinstance(out, tuple):
                return jnp.concatenate([v.ravel() for v in out])
            return out.ravel()

        pred = jax.vmap(_fwd)(flat)
        tgt_parts = [
            flat[:, layout.offsets[n] : layout.offsets[n] + layout.sizes[n]]
            for n in t.tgt
        ]
        tgt = jnp.concatenate(tgt_parts, axis=1)
        err = pred - tgt
        if curvatures is not None and t.id in curvatures:
            L = jnp.linalg.cholesky(curvatures[t.id])
            err = jnp.einsum("ji,bj->bi", L, err)  # L^T @ err
        blocks.append(err)
    return jnp.concatenate(blocks, axis=1)


def decompose(
    graph: Graph,
    state: State,
    eta: float,
    T: int,
    curvatures: dict[str, jax.Array] | None = None,
) -> dict[str, Any]:
    """Full spectral decomposition of the profiled energy.

    φ_T = ||b_⊥||² + Σ (1 − ηλ_i)^{2T} c_i²
        = coverage_gap + conditioning_penalty

    Returns dict with eigenvalues, eigenvectors, per-example coverage gap,
    conditioning penalty, predicted φ_T, spectral filter, and effective rank.
    """
    A = weighted_jacobian(graph, state, curvatures)  # (B, m, D)
    A_avg = jnp.mean(A, axis=0)  # (m, D)

    # SVD of batch-averaged weighted Jacobian
    V_full, sigma_full, Ut_full = jnp.linalg.svd(A_avg, full_matrices=False)

    # Threshold to actual column-space rank (clamped columns create zero singular values)
    tol = float(sigma_full[0]) * 1e-5 if len(sigma_full) > 0 else 0.0
    rank = int(jnp.sum(sigma_full > tol))
    V = V_full[:, :rank]  # (m, rank) — column-space basis
    eigenvalues = sigma_full[:rank] ** 2  # precision matrix eigenvalues
    eigenvectors = Ut_full[:rank, :].T  # (D, rank) — eigenvectors

    # Task residual
    b = task_residual(graph, state, curvatures)  # (B, m)

    # Project b onto column space of A (spanned by columns of V)
    c = b @ V  # (B, rank)
    b_perp = b - c @ V.T  # (B, m)

    coverage_gap = jnp.sum(b_perp**2, axis=1)  # (B,)

    # Spectral filter: ((1 - ηλ)²)^T
    spectral_filter = ((1.0 - eta * eigenvalues) ** 2) ** T  # (rank,)
    conditioning_penalty = jnp.sum(spectral_filter[None, :] * c**2, axis=1)  # (B,)

    phi_T_predicted = coverage_gap + conditioning_penalty

    gate = 1.0 / (eta * T) if T > 0 else float("inf")
    effective_rank = int(jnp.sum(eigenvalues > gate))

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "coefficients": c,
        "b_perp": b_perp,
        "coverage_gap": coverage_gap,
        "conditioning_penalty": conditioning_penalty,
        "phi_T_predicted": phi_T_predicted,
        "spectral_filter": spectral_filter,
        "effective_rank": effective_rank,
    }


def score(
    graph: Graph,
    state: State,
    eta: float,
    T: int,
    curvatures: dict[str, jax.Array] | None = None,
) -> dict[str, Any]:
    """Batch-averaged architecture score summary."""
    d = decompose(graph, state, eta, T, curvatures)
    eigs = d["eigenvalues"]
    nz = eigs[eigs > 1e-10]
    cond = float(nz[0] / nz[-1]) if len(nz) > 1 else 1.0
    return {
        "phi_T_predicted": float(jnp.mean(d["phi_T_predicted"])),
        "coverage_gap": float(jnp.mean(d["coverage_gap"])),
        "conditioning_penalty": float(jnp.mean(d["conditioning_penalty"])),
        "effective_rank": d["effective_rank"],
        "condition_number": cond,
        "eigenvalues": eigs,
    }
