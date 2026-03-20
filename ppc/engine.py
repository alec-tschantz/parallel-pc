"""Functional engine: init, forward, infer, energy, state_grad, param_grad, variable, transform."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .graph import Graph, VarArg, TransformArg
from .types import State


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def init(
    graph: Graph,
    clamps: dict[str, jax.Array],
    *,
    key: jax.Array,
) -> State:
    """Allocate flat buffer, write clamps, init free vars with small randn."""
    batch = next(iter(clamps.values())).shape[0]
    layout = graph.layout

    flat = 0.01 * jax.random.normal(key, (batch, layout.total_dim))
    free_mask = jnp.ones(layout.total_dim)
    for name, val in clamps.items():
        o, s = layout.offsets[name], layout.sizes[name]
        flat = flat.at[:, o : o + s].set(val.reshape(batch, s))
        free_mask = free_mask.at[o : o + s].set(0.0)

    return State(flat=flat, free_mask=free_mask)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def predict(graph: Graph, state: State) -> dict[str, jax.Array]:
    """Run all transform buckets. Returns {target_var_name: (batch, *shape)}."""
    flat = state.flat
    predictions: dict[str, jax.Array] = {}

    for bucket in graph.buckets:
        n_edges = len(bucket.meta.transform_indices)
        batch = flat.shape[0]

        sources = []
        for si in range(bucket.meta.n_srcs):
            gathered = flat[:, bucket.gather_indices[si]]
            src_shape = bucket.meta.src_shapes[si]
            gathered = gathered.reshape(batch, n_edges, *src_shape)
            sources.append(gathered.transpose(1, 0, *range(2, gathered.ndim)))

        def apply_one(params: Any, *src_args: jax.Array) -> jax.Array:
            module = eqx.combine(
                jax.tree.unflatten(bucket.param_treedef, jax.tree.leaves(params)),
                bucket.static_module,
            )
            return jax.vmap(module)(*src_args)

        out = jax.vmap(apply_one, in_axes=(0, *([0] * len(sources))))(
            bucket.stacked_params, *sources
        )

        if bucket.meta.n_tgts == 1:
            out = (out,)

        for ei, tidx in enumerate(bucket.meta.transform_indices):
            t = graph.transforms[tidx]
            for oi, tgt_name in enumerate(t.tgt):
                predictions[tgt_name] = out[oi][ei]

    return predictions


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------


def energy(graph: Graph, state: State) -> jax.Array:
    """Total energy at current state (scalar, summed over batch)."""
    flat = state.flat
    predictions = predict(graph, state)

    total: jax.Array = jnp.float32(0.0)
    for ce in graph.compiled_energies:
        args = []
        for spec in ce.arg_specs:
            if isinstance(spec, VarArg):
                val = flat[:, spec.offset : spec.offset + spec.size].reshape(
                    flat.shape[0], *spec.shape
                )
                args.append(val)
            elif isinstance(spec, TransformArg):
                args.append(predictions[spec.var_name])
        e_per_example = jax.vmap(ce.fn)(*args)
        total = total + jnp.sum(e_per_example)
    return total


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def state_grad(graph: Graph, state: State) -> jax.Array:
    """Gradient of energy w.r.t. flat buffer, masked so clamped nodes get zero gradient."""

    def energy_fn(flat: jax.Array) -> jax.Array:
        return energy(graph, State(flat=flat, free_mask=state.free_mask))

    grads = jax.grad(energy_fn)(state.flat)
    return grads * state.free_mask


def param_grad(graph: Graph, state: State) -> Graph:
    """Gradient of energy w.r.t. transform params at converged state."""

    @eqx.filter_grad
    def _grad(g: Graph) -> jax.Array:
        return energy(g, state)

    return _grad(graph)


# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------


def infer(
    graph: Graph,
    state: State,
    *,
    optimizer: optax.GradientTransformation,
    iters: int = 5,
) -> State:
    """Run inference steps via lax.scan. Creates fresh optimizer state internally."""
    free_mask = state.free_mask
    opt_state = optimizer.init(state.flat)

    def step(
        carry: tuple[jax.Array, Any], _: None
    ) -> tuple[tuple[jax.Array, Any], None]:
        flat, opt_state = carry
        s = State(flat=flat, free_mask=free_mask)
        grads = state_grad(graph, s)
        updates, opt_state = optimizer.update(grads, opt_state, flat)
        flat = optax.apply_updates(flat, updates)
        return (flat, opt_state), None  # type: ignore

    (flat, _), _ = jax.lax.scan(
        step, (state.flat, opt_state), None, length=iters
    )
    return State(flat=flat, free_mask=free_mask)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def variable(graph: Graph, state: State, name: str) -> jax.Array:
    """Read a variable from state. Returns (batch, *shape)."""
    layout = graph.layout
    o, s, sh = layout.offsets[name], layout.sizes[name], layout.shapes[name]
    return state.flat[:, o : o + s].reshape(state.flat.shape[0], *sh)


def transform(graph: Graph, name: str) -> eqx.Module:
    """Extract a transform's module by ID."""
    for t in graph.transforms:
        if t.id == name:
            return t.module
    raise KeyError(f"Transform '{name}' not found")
