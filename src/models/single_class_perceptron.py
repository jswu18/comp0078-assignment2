from typing import Any, Dict

import numpy as np

from src.models.kernels import BaseKernel
from src.models.model import Model


class SingleClassPerceptron(Model):
    def __init__(
        self, kernel: BaseKernel, kernel_kwargs: Dict[str, Any], w=None, x=None
    ):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.x: np.ndarray = x
        self.w: np.ndarray = w
        super().__init__()

    @staticmethod
    def _predict(w: np.ndarray, gram: np.ndarray) -> np.ndarray:
        """
        prediction for a single data point (across all experiments)

        :param gram: gram matrix
                     (number_features, number_points)
        :return: a prediction for the single input
        """
        return np.sign(w @ gram)

    @staticmethod
    def _compute_update(w: np.ndarray, y: np.ndarray, gram: np.ndarray) -> np.ndarray:
        """
        compute weight update for a single data point (across all experiments)
        :param w: weight matrix
                  (number_training_points)
        :param gram: gram matrix
                     (number_features, number_features)
        :param y: matrix of a single response
        :return: weight update for the weights pertaining to the single input
        """
        prediction = SingleClassPerceptron._predict(w, gram)

        return (y != prediction) * y

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        All input matrices will share the first M dimensions representing the different independent experiments
        that we want to train for. N_i will be the size of the ith dimension, and i = 1, 2, ..., M

        :param x: design matrix
                  (number_training_points, number_of_dimensions)
        :param y: matrix of responses, the response for all parameter trials will be the same
                  (number_training_points)
        :return:
        """
        if "number_of_epochs" in kwargs:
            number_of_epochs = kwargs["number_of_epochs"]
        else:
            number_of_epochs = 1

        self.x = x
        gram = self.kernel(x, **self.kernel_kwargs)

        self.w = np.zeros(y.shape)

        for _ in range(number_of_epochs):
            for i in range(1, len(self.w)):
                self.w[i] += self._compute_update(w=self.w, y=y[i], gram=gram[:, i])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        perceptron prediction
        :param x: design matrix
                  (number_of_points, number_of_dimensions)
        :return: predictions for x
                 (number_of_points, number_classes)
        """
        gram = self.kernel(self.x, x, **self.kernel_kwargs)
        return self._predict(self.w, gram)
