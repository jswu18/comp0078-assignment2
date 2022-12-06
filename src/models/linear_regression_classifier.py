import jax.numpy as jnp
import numpy as np

def train(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(x) @ y


def predict(w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x @ w)
