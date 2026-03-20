import jax
import jax.numpy as jnp


def mse_energy(pred, value):
    """Squared error between prediction and value (sum reduction, 0.5 factor)."""
    return 0.5 * jnp.sum((pred - value) ** 2)


def cross_entropy_energy(pred, target):
    """Cross-entropy: target is one-hot, pred is logits."""
    return -jnp.sum(target * jax.nn.log_softmax(pred))
