from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import jit


def predict(w: np.ndarray, gram: np.ndarray) -> jnp.ndarray:
    """
    prediction for a single data point (across all experiments)
    :param w: weight matrix
              (N_1,...,N_M, number_training_points, number_classes)
    :param gram: precomputed gram matrix for a single input vs all training points (across all experiments)
                 (N_1,...,N_M, number_training_points, number_test_points)
    :return: a prediction for the single input (N_1,...,N_M, number_classes)
    """
    return gram.T @ w


def _compute_update(w: np.ndarray, y: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """
    compute weight update for a single data point (across all experiments)
    :param w: weight matrix
              (N_1,...,N_M, number_training_points, number_classes)
    :param gram: precomputed gram matrix for a single input vs all training points (across all experiments)
                 (N_1,...,N_M, number_training_points, 1)
    :param y: matrix of a single response (across all experiments)
              (N_1,...,N_M, number_classes)
    :return: weight update for the weights pertaining to the single input (N_1,...,N_M, number_classes)
    """
    prediction = predict(w, gram)
    update_mask = jnp.multiply(y, prediction) <= 0
    return jnp.multiply(y, update_mask)


_jit_compute_update = jit(_compute_update)


def _alternative_compute_update(
    w: np.ndarray, y: np.ndarray, gram: np.ndarray
) -> np.ndarray:
    prediction = predict(w, gram)
    prediction_idx = jnp.argmax(prediction)
    actual_idx = jnp.argmax(y)
    update_mask = np.zeros(y.shape)
    if prediction_idx != actual_idx:
        update_mask[prediction_idx] = 1
        update_mask[actual_idx] = 1
    return jnp.multiply(y, update_mask)


def train(
    w: np.ndarray,
    gram: np.ndarray,
    y: np.ndarray,
    number_of_epochs: int,
    use_default_update_method: bool = True,
) -> np.ndarray:
    """
    Vectorised perceptron training by training multiple trials, parameters, etc. in parallel
    The perceptron is trained one data point at a time, the new weights depend on the weights of the previous
    step, however each trial, parameter is independent, thus for our experiments, we can perform this in parallel.

    All input matrices will share the first M dimensions representing the different independent experiments
    that we want to train for. N_i will be the size of the ith dimension, and i = 1, 2, ..., M

    :param w: initial model weights
    :param gram: precomputed gram matrix
                 (N_1,...,N_M, number_training_points, number_training_points)
    :param y: matrix of responses, the response for all parameter trials will be the same
              (N_1,...,N_M, number_training_points, number_classes)
    :param number_of_epochs: number of epochs to train model
    :param use_default_update_method: update method choice
    :return:
    """
    if use_default_update_method:
        update = _jit_compute_update
    else:
        update = _alternative_compute_update
    for _ in range(int(number_of_epochs)):
        for i in range(1, gram.shape[-2]):
            w[..., i, :] += update(w, y=y[..., i, :], gram=gram[..., i : i + 1])
    return w
