"""MNIST training with ppc."""

import os
import time
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tyro
from keras.datasets import mnist as mnist_data

import ppc


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    iters: int = 20
    infer_lr: float = 0.05
    train_lr: float = 5e-5
    eval_every: int = 5


def load_mnist():
    (x_tr, y_tr), (x_te, y_te) = mnist_data.load_data()
    x_tr = jnp.array(x_tr.reshape(-1, 784).astype("float32") / 255.0)
    x_te = jnp.array(x_te.reshape(-1, 784).astype("float32") / 255.0)
    return x_tr, jax.nn.one_hot(y_tr, 10), x_te, jax.nn.one_hot(y_te, 10)


def make_graph(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return ppc.Graph(
        variables=[
            ppc.Variable("image", (784,)),
            ppc.Variable("h1", (256,)),
            ppc.Variable("h2", (128,)),
            ppc.Variable("label", (10,)),
        ],
        transforms=[
            ppc.Transform("t1", eqx.nn.Linear(784, 256, key=k1), src="image", tgt="h1"),
            ppc.Transform("t2", eqx.nn.Linear(256, 128, key=k2), src="h1", tgt="h2"),
            ppc.Transform("t3", eqx.nn.Linear(128, 10, key=k3), src="h2", tgt="label"),
        ],
        energies=[
            ppc.Energy(ppc.mse_energy, args=["t1", "h1"]),
            ppc.Energy(ppc.mse_energy, args=["t2", "h2"]),
            ppc.Energy(ppc.mse_energy, args=["t3", "label"]),
        ],
    )


def make_train_step(infer_opt, train_opt, iters):
    @eqx.filter_jit
    def train_step(graph, opt_state, images, labels, key):
        state = ppc.init(graph, {"image": images, "label": labels}, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=iters)
        loss = ppc.energy(graph, state)
        grads = ppc.param_grad(graph, state)
        updates, opt_state = train_opt.update(
            eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(graph, eqx.is_array)
        )
        graph = eqx.apply_updates(graph, updates)
        return graph, opt_state, loss

    return train_step


def make_eval_step(infer_opt, iters):
    @eqx.filter_jit
    def eval_step(graph, images, key):
        state = ppc.init(graph, {"image": images}, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=iters * 2)
        return ppc.variable(graph, state, "label")

    return eval_step


def evaluate(graph, eval_step, x, y, bs, key):
    correct, total = 0, 0
    for i in range(0, x.shape[0] - bs + 1, bs):
        key, ek = jax.random.split(key)
        preds = eval_step(graph, x[i : i + bs], ek)
        correct += int(jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y[i : i + bs], -1)))
        total += bs
    return correct / total


def main(cfg: Config):
    key = jax.random.PRNGKey(cfg.seed)
    x_tr, y_tr, x_te, y_te = load_mnist()

    key, gk = jax.random.split(key)
    graph = make_graph(gk)

    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))

    train_step = make_train_step(infer_opt, train_opt, cfg.iters)
    eval_step = make_eval_step(infer_opt, cfg.iters)

    # warmup
    key, sk = jax.random.split(key)
    graph, opt_state, _ = train_step(
        graph, opt_state, x_tr[: cfg.batch_size], y_tr[: cfg.batch_size], sk
    )

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        total_loss, n = 0.0, 0

        key, ek = jax.random.split(key)
        perm = jax.random.permutation(ek, x_tr.shape[0])
        for i in range(0, x_tr.shape[0] - cfg.batch_size + 1, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            key, sk = jax.random.split(key)
            graph, opt_state, loss = train_step(
                graph, opt_state, x_tr[idx], y_tr[idx], sk
            )
            total_loss += float(loss)
            n += 1

        dt = time.perf_counter() - t0
        avg_loss = total_loss / n
        line = f"epoch {epoch:3d}/{cfg.epochs}  loss={avg_loss:8.2f}  ({dt:.1f}s)"

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            key, ek = jax.random.split(key)
            acc = evaluate(graph, eval_step, x_te, y_te, cfg.batch_size, ek)
            line += f"  test_acc={acc:.4f}"

        print(line)


if __name__ == "__main__":
    tyro.cli(main)
