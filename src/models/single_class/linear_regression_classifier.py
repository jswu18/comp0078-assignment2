import jax.numpy as jnp
import numpy as np

from src.models.single_class.model import Model


class LinearRegressionClassifier(Model):
    @staticmethod
    def fit_predict(x_train, y_train, x_test, **kwargs):
        w = np.linalg.pinv(x_train) @ y_train
        return jnp.sign(x_test @ w)
