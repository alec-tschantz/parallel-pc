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
    n_epochs: int = 100
    iters: int = 50
    infer_lr: float = 0.01
    train_lr: float = 1e-3
    eval_every: int = 10


def load_mnist():
    (x_tr, y_tr), (x_te, y_te) = mnist_data.load_data()
    x_tr = jnp.array(x_tr.reshape(-1, 784).astype("float32") / 255.0)
    x_te = jnp.array(x_te.reshape(-1, 784).astype("float32") / 255.0)
    return x_tr, jax.nn.one_hot(y_tr, 10), x_te, jax.nn.one_hot(y_te, 10)


def batches(x, y, bs, key):
    perm = jax.random.permutation(key, x.shape[0])
    x, y = x[perm], y[perm]
    for i in range(0, x.shape[0] - bs + 1, bs):
        yield x[i : i + bs], y[i : i + bs]


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
            ppc.Transform(
                "t1",
                eqx.nn.Sequential(
                    [eqx.nn.Linear(784, 256, key=k1), eqx.nn.Lambda(jax.nn.relu)]
                ),
                src="image",
                tgt="h1",
            ),
            ppc.Transform(
                "t2",
                eqx.nn.Sequential(
                    [eqx.nn.Linear(256, 128, key=k2), eqx.nn.Lambda(jax.nn.relu)]
                ),
                src="h1",
                tgt="h2",
            ),
            ppc.Transform("t3", eqx.nn.Linear(128, 10, key=k3), src="h2", tgt="label"),
        ],
        energies=[
            ppc.Energy(ppc.mse_energy, args=["t1", "h1"]),
            ppc.Energy(ppc.mse_energy, args=["t2", "h2"]),
            ppc.Energy(ppc.cross_entropy_energy, args=["t3", "label"]),
        ],
    )


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------


def make_train_step(infer_opt, train_opt, iters):
    @eqx.filter_jit
    def train_step(graph, opt_state, images, labels, key):
        state = ppc.init(graph, {"image": images, "label": labels}, key=key)
        state = ppc.infer(graph, state, optimizer=infer_opt, iters=iters)
        loss = ppc.energy(graph, state)
        grads = ppc.param_grad(graph, state)
        updates, opt_state = train_opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(graph, eqx.is_array),
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


def evaluate(graph, eval_step, x, y, batch_size, key):
    correct, total = 0, 0
    for i in range(0, x.shape[0] - batch_size + 1, batch_size):
        key, ek = jax.random.split(key)
        preds = eval_step(graph, x[i : i + batch_size], ek)
        correct += int(
            jnp.sum(jnp.argmax(preds, -1) == jnp.argmax(y[i : i + batch_size], -1))
        )
        total += batch_size
    return correct / total, correct, total


def main(cfg: Config):
    key = jax.random.PRNGKey(cfg.seed)

    print("Loading MNIST...")
    x_tr, y_tr, x_te, y_te = load_mnist()
    print(f"  train={x_tr.shape[0]}, test={x_te.shape[0]}\n")

    key, gk = jax.random.split(key)
    graph = make_graph(gk)

    infer_opt = optax.adam(cfg.infer_lr)
    train_opt = optax.adam(cfg.train_lr)
    opt_state = train_opt.init(eqx.filter(graph, eqx.is_array))

    train_step = make_train_step(infer_opt, train_opt, cfg.iters)
    eval_step = make_eval_step(infer_opt, cfg.iters)

    for epoch in range(1, cfg.n_epochs + 1):
        t_epoch = time.perf_counter()
        total_loss, n_batches = 0.0, 0

        key, ek = jax.random.split(key)
        for images, labels in batches(x_tr, y_tr, cfg.batch_size, ek):
            key, sk = jax.random.split(key)
            graph, opt_state, loss = train_step(graph, opt_state, images, labels, sk)
            total_loss += float(loss)
            n_batches += 1

        dt = time.perf_counter() - t_epoch
        print(
            f"epoch {epoch:3d}/{cfg.n_epochs} | loss {total_loss / n_batches:.4f} | {dt:.1f}s"
        )

        if epoch % cfg.eval_every == 0:
            key, ek = jax.random.split(key)
            test_acc, correct, total = evaluate(
                graph, eval_step, x_te, y_te, cfg.batch_size, ek
            )
            print(f"  >> test_acc {test_acc:.4f} ({correct}/{total})")

    key, ek = jax.random.split(key)
    test_acc, correct, total = evaluate(
        graph, eval_step, x_te, y_te, cfg.batch_size, ek
    )
    print(f"\nFinal test accuracy: {test_acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    tyro.cli(main)
