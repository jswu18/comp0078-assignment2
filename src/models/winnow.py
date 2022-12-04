from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import jit


def _predict(w: np.ndarray, x: np.ndarray) -> jnp.ndarray:
    """
    prediction for a single data point (across all experiments)
    :param w: weight matrix
              (N_1,...,N_M, number_features)
    :param x: input data point (across all experiments)
              (N_1,...,N_M, number_features)
    :return: winnow prediction
             (N_1,...,N_M)
    """
    return (
        (jnp.mean(jnp.multiply(w, x), axis=-1) - 1).clip(0, 1).astype(bool).astype(int)
    )


def _compute_update(w: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    compute weight update for a single data point (across all experiments)
    :param w: weight matrix
              (N_1,...,N_M, number_features)
    :param x: input data point (across all experiments)
                 (N_1,...,N_M, number_features)
    :param y: matrix of a responses (across all experiments)
              (N_1,...,N_M)
    :return: new weight matrix (N_1,...,N_M, number_features)
    """
    prediction = _predict(w, x)  # (N_1,...,N_M)
    return jnp.multiply(w, 2 ** jnp.multiply((y - prediction)[..., None], x))


def train(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, number_of_epochs: int
) -> np.ndarray:
    """
    Vectorised winnow training by training multiple trials, etc. in parallel
    The winnow is trained one data point at a time, the new weights depend on the weights of the previous
    step, however each trial is independent, thus for our experiments, we can perform this in parallel.

    All input matrices will share the first M dimensions representing the different independent experiments
    that we want to train for. N_i will be the size of the ith dimension, and i = 1, 2, ..., M

    :param w: initial weight matrix
              (N_1,...,N_M, number_features)
    :param x: design matrix
              (N_1,...,N_M, number_features, number_training_points)
    :param y: matrix of responses, the response for all parameter trials will be the same
              (N_1,...,N_M, number_training_points)
    :param number_of_epochs: number of epochs to train model
    :return:
    """
    jit_compute_update = jit(_compute_update)
    for _ in range(number_of_epochs):
        for i in range(1, x.shape[-1]):
            w = jit_compute_update(w, y=y[..., i], x=x[..., i])
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    winnow prediction
    :param w: weight matrix
              (N_1,...,N_M, number_features)
    :param x: design matrix
              (N_1,...,N_M, number_features, number_of_test_points)
    :return: predictions for test points
             (N_1,...,N_M, number_of_test_points)
    """
    return np.stack([_predict(w, x[..., i]) for i in range(x.shape[-1])], axis=-1)
