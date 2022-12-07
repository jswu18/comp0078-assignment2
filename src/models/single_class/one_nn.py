from sklearn.neighbors import KDTree
import numpy as np

from src.models.single_class.model import Model


class OneNN(Model):
    @staticmethod
    def _all_distances(a, b):
        """
        a is (N, D)
        b is (M, D)
        returns all euclidean norm distances between a and b (N, M)

        """
        return np.sqrt(((a.T[:, None] - b.T[:, :, None]) ** 2).sum(0)).T

    @staticmethod
    def fit_predict(x_train, y_train, x_test, **kwargs):
        idx = np.argmin(OneNN._all_distances(x_test, x_train), axis=1)
        return y_train[idx]
