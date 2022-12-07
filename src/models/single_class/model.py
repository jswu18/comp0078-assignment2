from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    def __init__(self):
        pass

    """
    Abstract model class
    """

    @abstractmethod
    def fit_predict(self, x_train, y_train, x_test, **kwargs):
        pass

