from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import jit


def _predict(w: np.ndarray, gram: np.ndarray) -> jnp.ndarray:
    """
    prediction for a single data point (across all experiments)
    :param w: weight matrix
              (number_parameters, N_1,...,N_M, number_training_points, number_classes)
    :param gram: precomputed gram matrix for a single input vs all training points (across all experiments)
                 (number_parameters, N_1,...,N_M, number_training_points)
    :return: a prediction for the single input (number_parameters, N_1,...,N_M, number_classes)
    """
    return jnp.sum(jnp.multiply(w, gram[..., None]), axis=-2)


def _compute_update(w: np.ndarray, y: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """
    compute weight update for a single data point (across all experiments)
    :param w: weight matrix
              (number_parameters, N_1,...,N_M, number_training_points, number_classes)
    :param gram: precomputed gram matrix for a single input vs all training points (across all experiments)
                 (number_parameters, N_1,...,N_M, number_training_points)
    :param y: matrix of a single response (across all experiments)
              (N_1,...,N_M, number_classes)
    :return: weight update for the weights pertaining to the single input (N_1,...,N_M, number_classes)
    """
    prediction = _predict(w, gram)

    # negative if prediction was incorrect across all parameters
    correct_prediction_indicator = jnp.multiply(y[None, ...], prediction)

    # zero if prediction was incorrect
    clipped_correct_prediction_indicator = correct_prediction_indicator.clip(0, 1)

    # mask weights to update
    update_mask = ~clipped_correct_prediction_indicator.astype(bool)
    return jnp.multiply(y[None, ...], update_mask)


def train(
    gram: np.ndarray, y: np.ndarray, number_of_epochs: int
) -> np.ndarray:
    """
    Vectorised perceptron training by training multiple trials, parameters, etc. in parallel
    The perceptron is trained one data point at a time, the new weights depend on the weights of the previous
    step, however each trial, parameter is independent, thus for our experiments, we can perform this in parallel.

    All input matrices will share the first M dimensions representing the different independent experiments
    that we want to train for. N_i will be the size of the ith dimension, and i = 1, 2, ..., M

    :param gram: precomputed gram matrix
                 (number_parameters, N_1,...,N_M, number_training_points, number_training_points)
    :param y: matrix of responses, the response for all parameter trials will be the same
              (N_1,...,N_M, number_training_points, number_classes)
    :param number_of_epochs: number of epochs to train model
    :return:
    """
    number_of_parameters = gram.shape[0]
    number_training_points = gram.shape[-1]
    number_classes = y.shape[1]
    w = np.zeros((number_of_parameters, number_training_points, number_classes))

    jit_compute_update = jit(_compute_update)
    for _ in range(number_of_epochs):
        for i in range(1, gram.shape[-2]):
            w[..., i, :] += jit_compute_update(w, y=y[..., i, :], gram=gram[..., i])
    return w


def predict(w: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """
    perceptron prediction
    :param w: weight matrix
              (number_parameters, N_1,...,N_M, number_training_points, number_classes)
    :param gram: precomputed gram matrix
                 (number_parameters, N_1,...,N_M, number_training_points, number_of_test_points)
    :return: predictions for test points
             (number_parameters, N_1,...,N_M, number_of_test_points, number_classes)
    """
    return np.stack([_predict(w, gram[..., i]) for i in range(gram.shape[-1])], axis=-2)
