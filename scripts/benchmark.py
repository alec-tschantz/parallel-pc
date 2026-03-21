"""Scaling benchmark for ppc. Runs in <1 minute.

Usage: uv run benchmark.py
"""

import os
import sys
import time
from datetime import datetime
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import ppc
from ppc.engine import state_grad
from ppc.types import State


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def make_chain(n_layers: int, dim: int, key):
    """Linear chain: x -> h1 -> ... -> y."""
    keys = jax.random.split(key, n_layers)
    names = ["x"] + [f"h{i}" for i in range(1, n_layers)] + ["y"]
    variables = [ppc.Variable(n, (dim,)) for n in names]
    transforms, energies = [], []
    for i in range(n_layers):
        tid = f"t{i}"
        transforms.append(
            ppc.Transform(
                tid,
                eqx.nn.Linear(dim, dim, key=keys[i]),
                src=names[i],
                tgt=names[i + 1],
            )
        )
        energies.append(ppc.Energy(ppc.mse_energy, args=[tid, names[i + 1]]))
    return ppc.Graph(variables=variables, transforms=transforms, energies=energies)


def make_wide(n_branches: int, dim: int, key):
    """Star graph: x -> h_i for each branch (all same structure = 1 bucket)."""
    keys = jax.random.split(key, n_branches)
    variables = [ppc.Variable("x", (dim,))]
    transforms, energies = [], []
    for i in range(n_branches):
        hname = f"h{i}"
        tid = f"t{i}"
        variables.append(ppc.Variable(hname, (dim,)))
        transforms.append(
            ppc.Transform(tid, eqx.nn.Linear(dim, dim, key=keys[i]), src="x", tgt=hname)
        )
        energies.append(ppc.Energy(ppc.mse_energy, args=[tid, hname]))
    return ppc.Graph(variables=variables, transforms=transforms, energies=energies)


class _ConcatLinear(eqx.Module):
    linear: eqx.nn.Linear

    def __call__(self, a, b):
        return self.linear(jnp.concatenate([a, b], axis=-1))


def make_multi_io(n_layers: int, dim: int, key):
    """Chain with 2-input transforms: t_i(h_{i-1}, h_i) -> h_{i+1}."""
    keys = jax.random.split(key, n_layers)
    names = [f"h{i}" for i in range(n_layers + 2)]
    variables = [ppc.Variable(n, (dim,)) for n in names]
    transforms, energies = [], []
    for i in range(n_layers):
        tid = f"t{i}"
        transforms.append(
            ppc.Transform(
                tid,
                _ConcatLinear(eqx.nn.Linear(dim * 2, dim, key=keys[i])),
                src=[names[i], names[i + 1]],
                tgt=names[i + 2],
            )
        )
        energies.append(ppc.Energy(ppc.mse_energy, args=[tid, names[i + 2]]))
    return ppc.Graph(variables=variables, transforms=transforms, energies=energies)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

ITERS = 20
N_TRIALS = 10


def time_it(fn, n_trials=N_TRIALS):
    """Time fn(). Returns (compile_ms, avg_ms)."""
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(jax.tree.leaves(result))
    compile_ms = (time.perf_counter() - t0) * 1000

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(jax.tree.leaves(result))
        times.append((time.perf_counter() - t0) * 1000)

    return compile_ms, sum(times) / len(times)


def time_train(train_fn, graph, opt_state, clamps, key, n_trials=N_TRIALS):
    """Time a train function, threading state. Returns (compile_ms, avg_ms)."""
    t0 = time.perf_counter()
    graph, opt_state, loss = train_fn(graph, opt_state, clamps, key)
    jax.block_until_ready(loss)
    compile_ms = (time.perf_counter() - t0) * 1000

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        graph, opt_state, loss = train_fn(graph, opt_state, clamps, key)
        jax.block_until_ready(loss)
        times.append((time.perf_counter() - t0) * 1000)

    return compile_ms, sum(times) / len(times)


# ---------------------------------------------------------------------------
# JIT'd component functions (created once, recompile per shape)
# ---------------------------------------------------------------------------


def make_jit_fns(infer_opt, train_opt):
    @eqx.filter_jit
    def jit_energy(g, s):
        return ppc.energy(g, s)

    @eqx.filter_jit
    def jit_state_grad(g, s):
        return state_grad(g, s)

    @eqx.filter_jit
    def jit_param_grad(g, s):
        return ppc.param_grad(g, s)

    @eqx.filter_jit
    def jit_infer(g, s):
        return ppc.infer(g, s, optimizer=infer_opt, iters=ITERS)

    @eqx.filter_jit
    def jit_train(g, os, cl, k):
        s = ppc.init(g, cl, key=k)
        s = ppc.infer(g, s, optimizer=infer_opt, iters=ITERS)
        loss = ppc.energy(g, s)
        grads = ppc.param_grad(g, s)
        updates, new_os = train_opt.update(
            eqx.filter(grads, eqx.is_array), os, eqx.filter(g, eqx.is_array)
        )
        return eqx.apply_updates(g, updates), new_os, loss

    return jit_energy, jit_state_grad, jit_param_grad, jit_infer, jit_train


# ---------------------------------------------------------------------------
# Component breakdown
# ---------------------------------------------------------------------------


def bench_components():
    """Per-component timing on a representative config."""
    print(f"\n{'=' * 95}")
    print(f"  COMPONENT BREAKDOWN (chain depth=3, dim=128, bs=64, iters={ITERS})")
    print(f"{'=' * 95}")

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    infer_opt = optax.adam(0.01)
    train_opt = optax.adam(1e-3)
    graph = make_chain(3, 128, k1)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))
    clamps = {
        "x": jax.random.normal(k2, (64, 128)),
        "y": jax.random.normal(k3, (64, 128)),
    }
    state = ppc.init(graph, clamps, key=k3)

    jit_energy, jit_state_grad, jit_param_grad, jit_infer, jit_train = make_jit_fns(
        infer_opt, train_opt
    )

    # Infer first so we have a converged state for param_grad
    inferred = jit_infer(graph, state)
    jax.block_until_ready(inferred.flat)

    components = [
        ("energy", lambda: jit_energy(graph, state)),
        ("state_grad", lambda: jit_state_grad(graph, state)),
        ("param_grad", lambda: jit_param_grad(graph, inferred)),
        ("infer", lambda: jit_infer(graph, state)),
        ("train_step", None),  # handled separately
    ]

    print(f"  {'component':20s} {'compile':>9s} {'avg':>9s} {'per_iter':>9s}")
    print(f"  {'-' * 50}")

    for name, fn in components:
        if name == "train_step":
            c, a = time_train(jit_train, graph, opt_state, clamps, k3)
        else:
            c, a = time_it(fn)
        per_iter = f"{a / ITERS:.3f}ms" if name in ("infer", "train_step") else ""
        print(f"  {name:20s} {c:8.1f}ms {a:8.2f}ms {per_iter:>9s}")

    # Throughput
    _, avg_train = time_train(jit_train, graph, opt_state, clamps, k3)
    samples_per_sec = 64 / (avg_train / 1000)
    print(f"\n  Throughput: {samples_per_sec:.0f} samples/sec (bs=64)")


# ---------------------------------------------------------------------------
# Convergence tracking
# ---------------------------------------------------------------------------


def bench_convergence():
    """Track energy over inference iterations."""
    print(f"\n{'=' * 95}")
    print(f"  CONVERGENCE (chain depth=3, dim=128, bs=64)")
    print(f"{'=' * 95}")

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    infer_opt = optax.adam(0.01)
    graph = make_chain(3, 128, k1)
    clamps = {
        "x": jax.random.normal(k2, (64, 128)),
        "y": jax.random.normal(k3, (64, 128)),
    }

    @eqx.filter_jit
    def infer_with_trace(g, s):
        free_mask = s.free_mask
        opt_state = infer_opt.init(s.flat)

        def step(carry: tuple[jax.Array, Any], _):
            flat, opt_state = carry
            st = State(flat=flat, free_mask=free_mask)
            grads = state_grad(g, st)
            e = ppc.energy(g, st)
            updates, opt_state = infer_opt.update(grads, opt_state, flat)
            flat = optax.apply_updates(flat, updates)
            return (flat, opt_state), e

        (flat, _), energies = jax.lax.scan(step, (s.flat, opt_state), None, length=50)  # type: ignore
        return energies

    state = ppc.init(graph, clamps, key=k3)
    energies = infer_with_trace(graph, state)
    energies = [float(e) for e in energies]

    print(f"  {'iter':>6s} {'energy':>12s} {'delta':>12s}")
    print(f"  {'-' * 32}")
    for i in [0, 1, 2, 5, 10, 20, 30, 49]:
        delta = f"{energies[i] - energies[i-1]:+.4f}" if i > 0 else ""
        print(f"  {i:6d} {energies[i]:12.4f} {delta:>12s}")


# ---------------------------------------------------------------------------
# Scaling suite
# ---------------------------------------------------------------------------


def bench_scaling(name, graph_fn, dims, batch_sizes):
    infer_opt = optax.adam(0.01)
    train_opt = optax.adam(1e-3)
    _, _, _, jit_infer, jit_train = make_jit_fns(infer_opt, train_opt)
    key = jax.random.PRNGKey(0)

    print(f"\n{'=' * 105}")
    print(f"  {name}")
    print(f"{'=' * 105}")
    header = (
        f"  {'config':28s} {'bkt':>4s} {'flat':>6s} "
        f"{'infer':>8s} {'train':>8s} {'ms/iter':>8s} {'samp/s':>8s} "
        f"{'eqns':>5s} {'scan':>4s}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for dim, bs in [(d, b) for d in dims for b in batch_sizes]:
        k1, k2, k3 = jax.random.split(key, 3)
        graph = graph_fn(dim=dim, key=k1)
        opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))

        var_names = list(graph.layout.var_names)
        first, last = var_names[0], var_names[-1]
        clamps = {
            first: jax.random.normal(k2, (bs, graph.layout.sizes[first])),
            last: jax.random.normal(k3, (bs, graph.layout.sizes[last])),
        }
        state = ppc.init(graph, clamps, key=k3)

        # Jaxpr
        jaxpr = jax.make_jaxpr(
            lambda s: ppc.infer(graph, s, optimizer=infer_opt, iters=ITERS)
        )(state)
        n_eqns = len(jaxpr.jaxpr.eqns)
        has_scan = any("scan" in str(e.primitive) for e in jaxpr.jaxpr.eqns)

        # Infer (JIT'd)
        _, avg_infer = time_it(lambda g=graph, s=state: jit_infer(g, s))

        # Train (JIT'd, threaded)
        _, avg_train = time_train(jit_train, graph, opt_state, clamps, k3)

        ms_per_iter = avg_infer / ITERS
        samples_per_sec = bs / (avg_train / 1000)

        config_str = f"dim={dim:3d} bs={bs:4d}"
        print(
            f"  {config_str:28s} {len(graph.buckets):4d} {graph.layout.total_dim:6d} "
            f"{avg_infer:7.1f}ms {avg_train:7.1f}ms {ms_per_iter:7.2f}ms "
            f"{samples_per_sec:7.0f}/s "
            f"{n_eqns:5d} {'yes' if has_scan else 'NO':>4s}"
        )


# ---------------------------------------------------------------------------
# Recompilation + memory
# ---------------------------------------------------------------------------


def bench_recompilation():
    print(f"\n{'=' * 105}")
    print(f"  RECOMPILATION CHECK")
    print(f"{'=' * 105}")

    key = jax.random.PRNGKey(0)
    graph = make_chain(3, 64, key)
    infer_opt = optax.adam(0.01)

    @eqx.filter_jit
    def do_infer(g, s):
        return ppc.infer(g, s, optimizer=infer_opt, iters=10)

    state = ppc.init(graph, {"x": jnp.ones((8, 64)), "y": jnp.ones((8, 64))}, key=key)
    jax.block_until_ready(do_infer(graph, state).flat)

    import io, logging

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("jax")
    logger.addHandler(handler)

    with jax.log_compiles():
        for i in range(5):
            k = jax.random.PRNGKey(i + 100)
            s = ppc.init(graph, {"x": jnp.ones((8, 64)), "y": jnp.ones((8, 64))}, key=k)
            jax.block_until_ready(do_infer(graph, s).flat)
        s_eval = ppc.init(graph, {"x": jnp.ones((8, 64))}, key=key)
        jax.block_until_ready(do_infer(graph, s_eval).flat)

    logger.removeHandler(handler)
    log_output = log_stream.getvalue()

    if log_output.strip():
        print(f"  WARNING: Recompilations detected:\n{log_output}")
    else:
        print("  OK: No recompilations (5 train + 1 eval, same shapes)")


def bench_memory():
    print(f"\n{'=' * 105}")
    print(f"  MEMORY (chain depth=3, dim=256, bs=64)")
    print(f"{'=' * 105}")

    key = jax.random.PRNGKey(0)
    graph = make_chain(3, 256, key)
    infer_opt = optax.adam(0.01)

    def kb(tree):
        return (
            sum(l.nbytes for l in jax.tree.leaves(tree) if hasattr(l, "nbytes")) / 1024
        )

    state = ppc.init(
        graph, {"x": jnp.ones((64, 256)), "y": jnp.ones((64, 256))}, key=key
    )
    opt_state = infer_opt.init(state.flat)

    print(f"  Graph params:       {kb(eqx.filter(graph, eqx.is_array)):8.1f} KB")
    print(f"  State flat buffer:  {kb(state.flat):8.1f} KB  {state.flat.shape}")
    print(f"  State free_mask:    {kb(state.free_mask):8.1f} KB")
    print(f"  Infer opt state:    {kb(opt_state):8.1f} KB  (created inside infer)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t_start = time.perf_counter()

    bench_components()
    bench_convergence()

    bench_scaling(
        "CHAIN (depth=3): batch + dim scaling",
        lambda dim, key: make_chain(3, dim, key),
        dims=[64, 128, 256],
        batch_sizes=[32, 128, 512],
    )

    bench_scaling(
        "CHAIN (dim=128): depth scaling",
        lambda dim, key: make_chain(dim, 128, key),
        dims=[3, 6, 12],
        batch_sizes=[64],
    )

    bench_scaling(
        "WIDE (dim=128): branch scaling (should stay 1 bucket)",
        lambda dim, key: make_wide(dim, 128, key),
        dims=[4, 16, 64],
        batch_sizes=[64],
    )

    bench_scaling(
        "MULTI-INPUT (2-ary, dim=128): depth scaling",
        lambda dim, key: make_multi_io(dim, 128, key),
        dims=[3, 6, 12],
        batch_sizes=[64],
    )

    bench_recompilation()
    bench_memory()

    total = time.perf_counter() - t_start
    print(f"\n  Total benchmark time: {total:.1f}s\n")


if __name__ == "__main__":
    # Tee output to outputs/ with timestamp
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"outputs/bench_{ts}.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(outpath, "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        main()
        sys.stdout = sys.__stdout__
    print(f"Saved to {outpath}")
