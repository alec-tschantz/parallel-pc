"""Inference precision matrix metrics and leverage-score reduction.

Γ_G = Σ_e J_e^T Λ_e J_e   (inference precision matrix)
φ_T = ||b_⊥||² + Σ (1-ηλ_i)^{2T} c_i²   (coverage gap + conditioning penalty)
ℓ_e = tr(Λ_e J_e M⁻¹ J_eᵀ)   (per-edge leverage score)
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
    """Jacobian of the residual (pred - target) for a single edge w.r.t. flat state.

    Returns shape (B, d_tgt, D).
    """
    t = graph.transforms[transform_idx]
    layout = graph.layout

    def residual(flat_single):
        srcs = []
        for s in t.src:
            o, sz, sh = layout.offsets[s], layout.sizes[s], layout.shapes[s]
            srcs.append(flat_single[o : o + sz].reshape(sh))
        out = t.module(*srcs)  # type: ignore
        if isinstance(out, tuple):
            pred = jnp.concatenate([v.ravel() for v in out])
        else:
            pred = out.ravel()
        tgt_parts = []
        for n in t.tgt:
            o, sz = layout.offsets[n], layout.sizes[n]
            tgt_parts.append(flat_single[o : o + sz])
        tgt = jnp.concatenate(tgt_parts)
        return pred - tgt

    J = jax.vmap(jax.jacrev(residual))(state.flat)
    return J * state.free_mask[None, None, :]


def weighted_jacobian(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Weighted state Jacobian A_G = [Λ_1^{1/2} J_1; ...; Λ_M^{1/2} J_M].

    Shape (B, m, D) where m = Σ d_tgt(e).
    """
    blocks = []
    for i, t in enumerate(graph.transforms):
        J = edge_jacobian(graph, state, i)
        if curvatures is not None and t.id in curvatures:
            L = jnp.linalg.cholesky(curvatures[t.id])
            J = jnp.einsum("ji,bjk->bik", L, J)
        blocks.append(J)
    return jnp.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Precision matrix and its inverse
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


def precision_inverse(
    graph: Graph,
    state: State,
    eps: float = 1e-4,
    curvatures: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """M⁻¹ = (Γ + εI)⁻¹. Shape (D, D). Computed once, used for all leverage scores."""
    G = precision_matrix(graph, state, curvatures)
    D = G.shape[0]
    M = G + eps * jnp.eye(D)
    return jnp.linalg.inv(M)


# ---------------------------------------------------------------------------
# Leverage scores (Section 3.2 of paper)
# ---------------------------------------------------------------------------


def leverage_scores(
    graph: Graph,
    state: State,
    M_inv: jax.Array,
    curvatures: dict[str, jax.Array] | None = None,
) -> list[float]:
    """Per-edge leverage scores: ℓ_e = tr(Λ_e J_e M⁻¹ J_eᵀ).

    High leverage = edge contributes unique directions to the column space.
    Low leverage = edge is redundant (other edges already cover its directions).

    M_inv: precomputed (D, D) from precision_inverse().
    Returns one float per edge.
    """
    scores = []
    for i, t in enumerate(graph.transforms):
        J = edge_jacobian(graph, state, i)
        J_avg = jnp.mean(J, axis=0)  # (d_tgt, D)
        if curvatures is not None and t.id in curvatures:
            # ℓ_e = tr(Λ_e J_e M⁻¹ J_eᵀ)
            A = J_avg @ M_inv  # (d_tgt, D)
            B = A @ J_avg.T  # (d_tgt, d_tgt)
            ell = float(jnp.trace(curvatures[t.id] @ B))
        else:
            # Λ_e = I: ℓ_e = tr(J_e M⁻¹ J_eᵀ)
            A = J_avg @ M_inv  # (d_tgt, D)
            ell = float(jnp.sum(A * J_avg))  # tr(A J^T) = sum(A ⊙ J)
        scores.append(ell)
    return scores


def woodbury_downdate(
    M_inv: jax.Array,
    J_e: jax.Array,
    Lambda_e_inv: jax.Array | None = None,
) -> jax.Array:
    """Update M⁻¹ after removing edge e's PSD contribution.

    (M - J_eᵀ Λ_e J_e)⁻¹ = M⁻¹ + M⁻¹ J_eᵀ (Λ_e⁻¹ - J_e M⁻¹ J_eᵀ)⁻¹ J_e M⁻¹

    J_e: (d_tgt, D) batch-averaged Jacobian.
    Lambda_e_inv: (d_tgt, d_tgt) inverse curvature, or None for Λ=I.
    Returns updated M⁻¹ of shape (D, D).
    """
    d = J_e.shape[0]
    A = M_inv @ J_e.T  # (D, d_tgt)
    JMJ = J_e @ A  # (d_tgt, d_tgt)
    if Lambda_e_inv is None:
        Lambda_e_inv = jnp.eye(d)
    S = jnp.linalg.inv(Lambda_e_inv - JMJ)  # (d_tgt, d_tgt)
    return M_inv + A @ S @ A.T


# ---------------------------------------------------------------------------
# Task residual and spectral decomposition
# ---------------------------------------------------------------------------


def task_residual(
    graph: Graph,
    state: State,
    curvatures: dict[str, jax.Array] | None = None,
) -> jax.Array:
    """Stacked curvature-weighted prediction errors. Shape (B, m)."""
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
            err = jnp.einsum("ji,bj->bi", L, err)
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
    """
    A = weighted_jacobian(graph, state, curvatures)
    A_avg = jnp.mean(A, axis=0)

    V_full, sigma_full, Ut_full = jnp.linalg.svd(A_avg, full_matrices=False)
    tol = float(sigma_full[0]) * 1e-5 if len(sigma_full) > 0 else 0.0
    rank = int(jnp.sum(sigma_full > tol))
    V = V_full[:, :rank]
    eigenvalues = sigma_full[:rank] ** 2
    eigenvectors = Ut_full[:rank, :].T

    b = task_residual(graph, state, curvatures)
    c = b @ V
    b_perp = b - c @ V.T

    coverage_gap = jnp.sum(b_perp**2, axis=1)
    spectral_filter = ((1.0 - eta * eigenvalues) ** 2) ** T
    conditioning_penalty = jnp.sum(spectral_filter[None, :] * c**2, axis=1)
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
