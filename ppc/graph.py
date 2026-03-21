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
class VarGather:
    """Gather from flat buffer for one arg position across all energies in a bucket."""

    gather_indices: tuple[tuple[int, ...], ...]  # (n_energies, size) as nested tuples
    shape: tuple[int, ...]


@dataclass(frozen=True)
class TransformGather:
    """Gather from predictions flat buffer for one arg position across all energies in a bucket."""

    var_names: tuple[str, ...]  # one per energy
    gather_indices: tuple[tuple[int, ...], ...]  # (n_energies, size) as nested tuples
    shape: tuple[int, ...]


@dataclass(frozen=True)
class EnergyBucket:
    """Energies with same fn and arg pattern, batched together."""

    fn: Callable
    n_energies: int
    arg_gathers: tuple[VarGather | TransformGather, ...]


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
    tgt_names: tuple[tuple[str, ...], ...]  # per-edge target variable names
    shared_srcs: tuple[
        bool, ...
    ]  # per-source: True if all edges share the same source var


class TransformBucket(eqx.Module):
    """Transforms with identical structure, vmapped together."""

    stacked_params: Any
    static_module: Any = eqx.field(static=True)
    param_treedef: Any = eqx.field(static=True)
    meta: BucketMeta = eqx.field(static=True)
    gather_indices: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    scatter_indices: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(
        static=True
    )  # per-tgt: (n_edges, size)


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
    energy_buckets: tuple[EnergyBucket, ...] = eqx.field(static=True)

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
        e_buckets = _bucket_energies(compiled, layout)

        self.variables = vars_
        self.transforms = txs_
        self.energies = ens_
        self.layout = layout
        self.buckets = tuple(buckets)
        self.compiled_energies = tuple(compiled)
        self.energy_buckets = tuple(e_buckets)


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
        gather_indices = tuple(
            tuple(tuple(int(x) for x in row) for row in np.stack(g))
            for g in gather_per_src
        )

        tgt_specs = []
        scatter_per_tgt: list[list] = [[] for _ in range(len(t0.tgt))]
        for t in group:
            tgt_specs.append(
                tuple(
                    (layout.offsets[n], layout.sizes[n], layout.shapes[n])
                    for n in t.tgt
                )
            )
            for ti, n in enumerate(t.tgt):
                o, sz = layout.offsets[n], layout.sizes[n]
                scatter_per_tgt[ti].append(np.arange(o, o + sz))
        scatter_indices = tuple(
            tuple(tuple(int(x) for x in row) for row in np.stack(g))
            for g in scatter_per_tgt
        )

        # Detect shared sources: all edges read same var for this src position
        shared_srcs = tuple(
            len(set(t.src[si] for t in group)) == 1 for si in range(len(t0.src))
        )

        meta = BucketMeta(
            src_specs=tuple(src_specs),
            tgt_specs=tuple(tgt_specs),
            transform_indices=tuple(indices),
            n_srcs=len(t0.src),
            n_tgts=len(t0.tgt),
            src_shapes=tuple(layout.shapes[s] for s in t0.src),
            tgt_shapes=tuple(layout.shapes[tgt] for tgt in t0.tgt),
            tgt_names=tuple(t.tgt for t in group),
            shared_srcs=shared_srcs,
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
                    scatter_indices=scatter_indices,
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


def _bucket_energies(
    compiled: list[CompiledEnergy],
    layout: Layout,
) -> list[EnergyBucket]:
    """Group compiled energies by fn + arg pattern for batched evaluation."""

    # Group by (fn, n_args, per-arg type tag + shape)
    def bucket_key(ce: CompiledEnergy) -> tuple:
        arg_sig = []
        for spec in ce.arg_specs:
            if isinstance(spec, VarArg):
                arg_sig.append(("var", spec.shape))
            else:
                arg_sig.append(("transform",))
        return (id(ce.fn), tuple(arg_sig))

    groups: dict[tuple, list[int]] = {}
    for i, ce in enumerate(compiled):
        key = bucket_key(ce)
        groups.setdefault(key, []).append(i)

    buckets = []
    for indices in groups.values():
        group = [compiled[i] for i in indices]
        ce0 = group[0]
        n_args = len(ce0.arg_specs)
        n_energies = len(group)

        arg_gathers: list[VarGather | TransformGather] = []
        for ai in range(n_args):
            spec0 = ce0.arg_specs[ai]
            if isinstance(spec0, VarArg):
                # Build gather indices: (n_energies, size) — each row is flat-buffer indices
                gather_rows = []
                for ce in group:
                    s = ce.arg_specs[ai]
                    assert isinstance(s, VarArg)
                    gather_rows.append(tuple(range(s.offset, s.offset + s.size)))
                arg_gathers.append(
                    VarGather(
                        gather_indices=tuple(gather_rows),
                        shape=spec0.shape,
                    )
                )
            else:
                assert isinstance(spec0, TransformArg)
                var_names_list = []
                gather_rows = []
                for ce in group:
                    s = ce.arg_specs[ai]
                    assert isinstance(s, TransformArg)
                    vn = s.var_name
                    var_names_list.append(vn)
                    o, sz = layout.offsets[vn], layout.sizes[vn]
                    gather_rows.append(tuple(range(o, o + sz)))
                arg_gathers.append(
                    TransformGather(
                        var_names=tuple(var_names_list),
                        gather_indices=tuple(gather_rows),
                        shape=layout.shapes[var_names_list[0]],
                    )
                )

        buckets.append(
            EnergyBucket(
                fn=ce0.fn,
                n_energies=n_energies,
                arg_gathers=tuple(arg_gathers),
            )
        )

    return buckets


def expand(
    graph: Graph,
    new_transforms: Sequence[Transform] = (),
    new_energies: Sequence[Energy] = (),
    new_variables: Sequence[Variable] = (),
) -> Graph:
    """Return a new Graph with additional variables, transforms, and energies."""
    return Graph(
        variables=list(graph.variables) + list(new_variables),
        transforms=list(graph.transforms) + list(new_transforms),
        energies=list(graph.energies) + list(new_energies),
    )
