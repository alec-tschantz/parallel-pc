"""Inference precision matrix, boundary evaluation, and leverage-score reduction.

Γ_G = Σ_e J_e^T Λ_e J_e         (inference precision matrix, all edges)
φ_T^B = ||b_B^⊥||² + Σ (1-ηλ_i)^{2T} c_{B,i}²   (boundary profiled energy)
ℓ̃_e = tr(J_e M⁻¹ J_eᵀ)         (trace leverage, ranking proxy)
"""

from __future__ import annotations

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
    layout = graph.layout
    free_mask = state.free_mask
    boundary, internal = [], []
    for i, t in enumerate(graph.transforms):
        is_boundary = False
        for name in list(t.src) + list(t.tgt):
            o = layout.offsets[name]
            if float(free_mask[o]) == 0.0:
                is_boundary = True
                break
        if is_boundary:
            boundary.append(i)
        else:
            internal.append(i)
    return {"boundary": boundary, "internal": internal}


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


def edge_jacobian(graph: Graph, state: State, transform_idx: int) -> jax.Array:
    """Jacobian of residual (pred - target) for edge w.r.t. flat state. (B, d_tgt, D)."""
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
        return (
            jnp.concatenate(tgt_parts) - pred
        )  # note: tgt - pred (positive when undershoot)

    J = jax.vmap(jax.jacrev(residual))(state.flat)
    return J * state.free_mask[None, None, :]


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def task_residual(graph: Graph, state: State) -> jax.Array:
    """Full task residual (all edges). Shape (B, m)."""
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
        blocks.append(pred - tgt)
    return jnp.concatenate(blocks, axis=1)


def boundary_residual(graph: Graph, state: State) -> jax.Array:
    """Task residual restricted to boundary edges. Shape (B, m_B)."""
    layout = graph.layout
    flat = state.flat
    edges = classify_edges(graph, state)
    blocks = []
    for i in edges["boundary"]:
        t = graph.transforms[i]

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
        blocks.append(pred - tgt)
    if not blocks:
        return jnp.zeros((flat.shape[0], 0))
    return jnp.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Precision matrix
# ---------------------------------------------------------------------------


def edge_precision(graph: Graph, state: State) -> list[jax.Array]:
    """Per-edge PSD contributions J_e^T J_e, each (D, D), averaged over batch."""
    terms = []
    for i in range(len(graph.transforms)):
        J = edge_jacobian(graph, state, i)
        gram = jnp.einsum("bji,bjk->bik", J, J)
        terms.append(jnp.mean(gram, axis=0))
    return terms


def precision_matrix(graph: Graph, state: State) -> jax.Array:
    """Γ = Σ J_e^T J_e. Shape (D, D), batch-averaged."""
    return sum(edge_precision(graph, state))  # type: ignore


def precision_inverse(graph: Graph, state: State, eps: float = 1e-4) -> jax.Array:
    """M⁻¹ = (Γ + εI)⁻¹. Shape (D, D)."""
    G = precision_matrix(graph, state)
    return jnp.linalg.inv(G + eps * jnp.eye(G.shape[0]))


# ---------------------------------------------------------------------------
# Leverage scores (trace proxy for ranking)
# ---------------------------------------------------------------------------


def leverage_scores(
    graph: Graph, state: State, M_inv: jax.Array, edge_indices: list[int] | None = None
) -> dict[int, float]:
    """Trace leverage ℓ̃_e = tr(J_e M⁻¹ J_eᵀ) for specified edges.

    Returns {edge_index: leverage}. If edge_indices is None, computes for all.
    """
    if edge_indices is None:
        edge_indices = list(range(len(graph.transforms)))
    scores = {}
    for i in edge_indices:
        J = edge_jacobian(graph, state, i)
        J_avg = jnp.mean(J, axis=0)  # (d_tgt, D)
        A = J_avg @ M_inv
        scores[i] = float(jnp.sum(A * J_avg))
    return scores


def woodbury_downdate(M_inv: jax.Array, J_e: jax.Array) -> jax.Array:
    """M⁻¹ after removing edge e: (M - J_eᵀ J_e)⁻¹ via Woodbury."""
    d = J_e.shape[0]
    A = M_inv @ J_e.T  # (D, d)
    JMJ = J_e @ A  # (d, d)
    S = jnp.linalg.inv(jnp.eye(d) - JMJ)  # (d, d)
    return M_inv + A @ S @ A.T


# ---------------------------------------------------------------------------
# Frozen-RHS boundary profiled energy (exact, for stopping criterion)
# ---------------------------------------------------------------------------


def frozen_boundary_phi(
    Gamma_reduced: jax.Array,
    b_B: jax.Array,
    A_B: jax.Array,
    d_frozen: jax.Array,
    eta: float,
    T: int,
) -> dict[str, Any]:
    """Boundary profiled energy with frozen driving force.

    Gamma_reduced: (D, D) precision matrix of reduced graph
    b_B: (B, m_B) boundary residual from full graph (frozen)
    A_B: (m_B, D) boundary Jacobian (batch-averaged, frozen)
    d_frozen: (D,) frozen driving force = A^T b (batch-averaged, from full graph)
    eta, T: inference parameters

    Returns dict with phi_T_B, coverage_gap, conditioning_penalty.
    """
    # Eigendecompose reduced Γ
    eigenvalues, eigenvectors = jnp.linalg.eigh(Gamma_reduced)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)  # numerical safety

    # Spectral filter g_T(λ) = (1-ηλ)^T applied to frozen driving force
    # State update: δx = U diag(g_i) U^T d  where g_i = 1 - (1-ηλ_i)^T / ...
    # Actually: after T steps of GD on quadratic E = ½||Ax-b||²:
    # x^(T) = U diag(1-(1-ηλ_i)^T) Λ⁻¹ U^T A^T b
    # = U diag((1-(1-ηλ_i)^T)/λ_i) U^T d_frozen
    # The remaining residual in boundary space:
    # b_B^(T) = b_B - A_B δx^(T)

    # Project d_frozen into eigenbasis
    d_eig = eigenvectors.T @ d_frozen  # (D,)

    # Spectral response: how much of d is resolved along each eigendirection
    # resolved_i = (1 - (1-ηλ_i)^T) / λ_i * d_eig_i  (for λ_i > 0)
    # = (1 - (1-ηλ_i)^T) / λ_i * d_eig_i
    safe_eigs = jnp.maximum(eigenvalues, 1e-10)
    filter_coeff = (1.0 - (1.0 - eta * eigenvalues) ** T) / safe_eigs
    filter_coeff = jnp.where(eigenvalues > 1e-10, filter_coeff, 0.0)

    # State correction in eigenbasis
    dx_eig = filter_coeff * d_eig  # (D,)
    dx = eigenvectors @ dx_eig  # (D,)

    # Boundary residual after correction
    b_B_avg = jnp.mean(b_B, axis=0)  # (m_B,)
    b_B_corrected = b_B_avg - A_B @ dx  # (m_B,)

    phi_T_B = float(jnp.sum(b_B_corrected**2))

    # Decompose into coverage gap + conditioning penalty
    # Project b_B into col(A_B)
    U_B, s_B, _ = jnp.linalg.svd(A_B, full_matrices=False)
    tol = float(s_B[0]) * 1e-5 if len(s_B) > 0 else 0.0
    rank_B = int(jnp.sum(s_B > tol))
    V_B = U_B[:, :rank_B]
    c_B = b_B_avg @ V_B
    b_B_perp = b_B_avg - c_B @ V_B.T
    coverage_gap = float(jnp.sum(b_B_perp**2))
    conditioning_penalty = phi_T_B - coverage_gap

    return {
        "phi_T_B": phi_T_B,
        "coverage_gap": coverage_gap,
        "conditioning_penalty": conditioning_penalty,
    }


# ---------------------------------------------------------------------------
# Full decompose (backwards compat for training evaluation)
# ---------------------------------------------------------------------------


def decompose(graph: Graph, state: State, eta: float, T: int) -> dict[str, Any]:
    """Full spectral decomposition of profiled energy (all edges)."""
    # Weighted Jacobian
    blocks = []
    for i in range(len(graph.transforms)):
        blocks.append(edge_jacobian(graph, state, i))
    A = jnp.mean(jnp.concatenate(blocks, axis=1), axis=0)  # (m, D)

    V_full, sigma_full, Ut_full = jnp.linalg.svd(A, full_matrices=False)
    tol = float(sigma_full[0]) * 1e-5 if len(sigma_full) > 0 else 0.0
    rank = int(jnp.sum(sigma_full > tol))
    V = V_full[:, :rank]
    eigenvalues = sigma_full[:rank] ** 2

    b = task_residual(graph, state)
    b_avg = jnp.mean(b, axis=0)
    c = b_avg @ V
    b_perp = b_avg - c @ V.T

    coverage_gap = float(jnp.sum(b_perp**2))
    spectral_filter = ((1.0 - eta * eigenvalues) ** 2) ** T
    conditioning_penalty = float(jnp.sum(spectral_filter * c**2))
    phi_T = coverage_gap + conditioning_penalty

    gate = 1.0 / (eta * T) if T > 0 else float("inf")
    effective_rank = int(jnp.sum(eigenvalues > gate))

    return {
        "phi_T": phi_T,
        "coverage_gap": coverage_gap,
        "conditioning_penalty": conditioning_penalty,
        "eigenvalues": eigenvalues,
        "effective_rank": effective_rank,
    }
