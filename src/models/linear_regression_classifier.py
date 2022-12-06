import jax.numpy as jnp
import numpy as np

from src.models.model import Model


class LinearRegressionClassifier(Model):
    def __init__(self, w=None):
        self.w: np.ndarray = w
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.w = np.linalg.pinv(x.T) @ y

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sign(x.T @ self.w)
