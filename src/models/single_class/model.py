from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    def __init__(self):
        pass

    """
    Abstract model class
    """

    @abstractmethod
    def fit_predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, **kwargs
    ) -> np.ndarray:
        pass
