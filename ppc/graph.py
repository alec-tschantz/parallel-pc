import math
import warnings
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable, NamedTuple

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

from .types import Energy, Transform, Variable


@dataclass(frozen=True)
class VarArg:
    """Energy arg -> variable slice in flat buffer."""

    offset: int
    size: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class TransformArg:
    """Energy arg -> transform output, keyed by target variable name."""

    var_name: str


@dataclass(frozen=True)
class CompiledEnergy:
    fn: Callable
    arg_specs: tuple[VarArg | TransformArg, ...]


@dataclass(frozen=True)
class Layout:
    total_dim: int
    var_names: tuple[str, ...]
    offsets: dict[str, int]
    sizes: dict[str, int]
    shapes: dict[str, tuple[int, ...]]


class BucketMeta(NamedTuple):
    """Static metadata for a transform bucket."""

    src_specs: tuple
    tgt_specs: tuple
    transform_indices: tuple[int, ...]
    n_srcs: int
    n_tgts: int
    src_shapes: tuple[tuple[int, ...], ...]
    tgt_shapes: tuple[tuple[int, ...], ...]


class TransformBucket(eqx.Module):
    """Transforms with identical structure, vmapped together."""

    stacked_params: Any
    static_module: Any = eqx.field(static=True)
    param_treedef: Any = eqx.field(static=True)
    meta: BucketMeta = eqx.field(static=True)
    gather_indices: tuple = eqx.field(static=True)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class Graph(eqx.Module):
    variables: tuple[Variable, ...] = eqx.field(static=True)
    transforms: tuple[Transform, ...]
    energies: tuple[Energy, ...] = eqx.field(static=True)
    layout: Layout = eqx.field(static=True)
    buckets: tuple[TransformBucket, ...]
    compiled_energies: tuple[CompiledEnergy, ...] = eqx.field(static=True)

    def __init__(
        self,
        variables: list[Variable] | tuple[Variable, ...],
        transforms: list[Transform] | tuple[Transform, ...],
        energies: list[Energy] | tuple[Energy, ...],
    ):
        vars_ = tuple(variables)
        txs_ = tuple(transforms)
        ens_ = tuple(energies)

        var_names = {v.name for v in vars_}
        transform_ids = {t.id for t in txs_}
        assert len(var_names) == len(vars_), "Duplicate variable names"
        assert len(transform_ids) == len(txs_), "Duplicate transform IDs"
        for t in txs_:
            for s in t.src:
                assert s in var_names, f"Transform '{t.id}' src '{s}' not found"
            for tgt in t.tgt:
                assert tgt in var_names, f"Transform '{t.id}' tgt '{tgt}' not found"

        layout = _build_layout(vars_)
        transform_id_to_idx = {t.id: i for i, t in enumerate(txs_)}

        buckets = _bucket_transforms(txs_, layout)
        compiled = _compile_energies(ens_, txs_, transform_id_to_idx, var_names, layout)

        self.variables = vars_
        self.transforms = txs_
        self.energies = ens_
        self.layout = layout
        self.buckets = tuple(buckets)
        self.compiled_energies = tuple(compiled)


def _build_layout(variables: tuple[Variable, ...]) -> Layout:
    offsets, sizes, shapes = {}, {}, {}
    offset = 0
    for v in variables:
        s = math.prod(v.shape)
        offsets[v.name] = offset
        sizes[v.name] = s
        shapes[v.name] = v.shape
        offset += s
    return Layout(
        total_dim=offset,
        var_names=tuple(v.name for v in variables),
        offsets=offsets,
        sizes=sizes,
        shapes=shapes,
    )


def _bucket_key(transform: Transform, layout: Layout):
    params, _ = eqx.partition(transform.module, eqx.is_array)
    treedef = jax.tree.structure(params)
    param_shapes = tuple(l.shape for l in jax.tree.leaves(params))
    src_shapes = tuple(layout.shapes[s] for s in transform.src)
    tgt_shapes = tuple(layout.shapes[t] for t in transform.tgt)
    return (
        treedef,
        param_shapes,
        len(transform.src),
        src_shapes,
        len(transform.tgt),
        tgt_shapes,
    )


def _bucket_transforms(
    transforms: tuple[Transform, ...], layout: Layout
) -> list[TransformBucket]:
    groups: dict[Any, list[int]] = {}
    for i, t in enumerate(transforms):
        key = _bucket_key(t, layout)
        groups.setdefault(key, []).append(i)

    buckets = []
    for indices in groups.values():
        group = [transforms[i] for i in indices]
        t0 = group[0]

        params_list = [eqx.partition(t.module, eqx.is_array)[0] for t in group]
        _, static = eqx.partition(t0.module, eqx.is_array)

        stacked = jax.tree.map(lambda *ps: jnp.stack(ps), *params_list)
        treedef = jax.tree.structure(params_list[0])

        src_specs = []
        gather_per_src = [[] for _ in range(len(t0.src))]
        for t in group:
            edge_srcs = []
            for si, s in enumerate(t.src):
                o, sz, sh = layout.offsets[s], layout.sizes[s], layout.shapes[s]
                edge_srcs.append((o, sz, sh))
                gather_per_src[si].append(np.arange(o, o + sz))
            src_specs.append(tuple(edge_srcs))
        gather_indices = tuple(np.stack(g) for g in gather_per_src)

        tgt_specs = []
        for t in group:
            tgt_specs.append(
                tuple(
                    (layout.offsets[n], layout.sizes[n], layout.shapes[n])
                    for n in t.tgt
                )
            )

        meta = BucketMeta(
            src_specs=tuple(src_specs),
            tgt_specs=tuple(tgt_specs),
            transform_indices=tuple(indices),
            n_srcs=len(t0.src),
            n_tgts=len(t0.tgt),
            src_shapes=tuple(layout.shapes[s] for s in t0.src),
            tgt_shapes=tuple(layout.shapes[tgt] for tgt in t0.tgt),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buckets.append(
                TransformBucket(
                    stacked_params=stacked,
                    static_module=static,
                    param_treedef=treedef,
                    meta=meta,
                    gather_indices=gather_indices,
                )
            )

    return buckets


def _compile_energies(
    energies: tuple[Energy, ...],
    transforms: tuple[Transform, ...],
    transform_id_to_idx: dict[str, int],
    var_names: set[str],
    layout: Layout,
) -> list[CompiledEnergy]:
    compiled = []
    for e in energies:
        specs = []
        for arg_str in e.args:
            if "." in arg_str:
                tid, tgt_name = arg_str.split(".", 1)
                assert tid in transform_id_to_idx, f"Transform '{tid}' not found"
                specs.append(TransformArg(tgt_name))
            elif arg_str in transform_id_to_idx:
                t = transforms[transform_id_to_idx[arg_str]]
                assert (
                    len(t.tgt) == 1
                ), f"Transform '{arg_str}' has multiple outputs, use '{arg_str}.target_name'"
                specs.append(TransformArg(t.tgt[0]))
            elif arg_str in var_names:
                specs.append(
                    VarArg(
                        layout.offsets[arg_str],
                        layout.sizes[arg_str],
                        layout.shapes[arg_str],
                    )
                )
            else:
                raise ValueError(
                    f"Energy arg '{arg_str}' not found as variable or transform"
                )
        compiled.append(CompiledEnergy(e.fn, tuple(specs)))
    return compiled
