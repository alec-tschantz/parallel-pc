# ppc

A JAX library for energy-based inference on arbitrary compute graphs. Define variables (mutable tensors), transforms (any Equinox module), and energy functions — `ppc` compiles this into a flat buffer with vmapped transform bucketing, then runs inference via `lax.scan`

## Install

```bash
uv sync
```

## Example

```python
import ppc
import jax, optax, equinox as eqx
from equinox.nn import Linear


key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)

data = (
    jax.random.normal(key, (32, 4)),
    jax.nn.one_hot(jax.numpy.zeros(32, dtype=int), 2),
)

graph = ppc.Graph(
    variables=[
        ppc.Variable("x", (4,)),
        ppc.Variable("h", (8,)),
        ppc.Variable("y", (2,)),
    ],
    transforms=[
        ppc.Transform("t1", Linear(4, 8, key=k1), src="x", tgt="h"),
        ppc.Transform("t2", Linear(8, 2, key=k2), src="h", tgt="y"),
    ],
    energies=[
        ppc.Energy(ppc.mse_energy, args=["t1", "h"]),
        ppc.Energy(ppc.cross_entropy_energy, args=["t2", "y"]),
    ],
)

infer_opt = optax.adam(0.01)
train_opt = optax.adam(1e-3)
opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))


for x, y in data:
    state = ppc.init(graph, {"x": x, "y": y}, key=key)
    state = ppc.infer(graph, state, optimizer=infer_opt, iters=5)

    loss = ppc.energy(graph, state)
    grads = ppc.param_grad(graph, state)
    updates, opt_state = train_opt.update(eqx.filter(grads, eqx.is_array), opt_state)
    graph = eqx.apply_updates(graph, updates)

```
