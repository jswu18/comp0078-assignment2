import jax.numpy as jnp


def train(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.solve(x.T @ x, x.T @ y)


def predict(w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x @ w)
