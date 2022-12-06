from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    def __init__(self):
        pass

    """
    Abstract model class
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fitting model
        :param x: dataset of shape (number of dimensions, number of examples)
        :param y: labels of shape (number of examples, 1)
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prediction of x
        :param x: dataset of shape (number of dimensions, number of examples)
        :return: predicted labels of shape (number of examples, 1)
        """
        pass
