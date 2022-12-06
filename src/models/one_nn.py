import numpy as np
from sklearn.neighbors import KDTree

from src.models.model import Model


class OneNN(Model):
    k = 1

    def __init__(self, tree=None, y=None):
        self.tree: KDTree = tree
        self.y: np.ndarray = y
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.tree = KDTree(x)
        self.y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.y[self.tree.query(x, k=self.k)[1]].reshape(
            -1,
        )
