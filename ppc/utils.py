import jax
import jax.numpy as jnp


def mse_energy(pred, value):
    """Mean squared error between prediction and value."""
    return jnp.mean((pred - value) ** 2)


def cross_entropy_energy(pred, target):
    """Cross-entropy: target is one-hot, pred is logits."""
    return -jnp.sum(target * jax.nn.log_softmax(pred))
