from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple


class Perceptron:
    @staticmethod
    def train(w: np.ndarray, gram: np.ndarray, y: np.ndarray, number_of_epochs: int) -> np.ndarray:
        """

        :param w:
        :param gram:
        :param y:
        :param number_of_epochs:
        :return:
        """
        for _ in range(number_of_epochs):
            for i in range(1, gram.shape[0]):
                # make prediction
                prediction = w.T @ gram[i, :]

                # negative if prediction was incorrect
                correct_prediction_indicator = jnp.multiply(y[i], prediction)

                # zero if prediction was incorrect
                clipped_correct_prediction_indicator = correct_prediction_indicator.clip(0, 1)

                # mask weights to update
                update_mask = ~clipped_correct_prediction_indicator.astype(bool)

                # update weights
                w = w.at[i, :].set(w[i, :] + jnp.multiply(y[i], update_mask))
        return w

    @staticmethod
    def predict(w: np.ndarray, gram: np.ndarray) -> np.ndarray:
        """

        :param w:
        :param gram:
        :return:
        """
        return w.T @ gram
