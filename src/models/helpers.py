from dataclasses import dataclass
from typing import List

import jax.numpy as jnp
import numpy as np
from sklearn.neighbors import KDTree


@dataclass
class TrainTestData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def x(self):
        return np.concatenate((self.x_train, self.x_test))

    @property
    def y(self):
        return np.concatenate((self.y_train, self.y_test))


def _build_train_test_data(x, y, indices):
    return TrainTestData(
        x_train=np.delete(x, indices, axis=0),
        y_train=np.delete(y, indices, axis=0),
        x_test=x[indices, :],
        y_test=y[indices, :],
    )


def split_train_test_data(
    x: np.ndarray, y: np.ndarray, train_percentage: float
) -> TrainTestData:
    n, _ = x.shape
    test_indices = np.random.choice(
        np.arange(n), int(np.round((1 - train_percentage) * n)), replace=False
    )
    return _build_train_test_data(x, y, indices=test_indices)


def make_folds(
    x: np.ndarray, y: np.ndarray, number_of_folds: int
) -> List[TrainTestData]:
    n, _ = x.shape
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, number_of_folds)
    return [
        _build_train_test_data(x, y, indices=split_index)
        for split_index in split_indices
    ]


def one_hot_encode(shifted_labels: np.ndarray) -> np.ndarray:
    """
    one hot encoding with (-1 and 1) of a vector of labels
    :param shifted_labels: labels vector shifted so that the min label is zero (number_data_points, )
    :return: one hot encoding of dimension (number_data_points, number_of_classes)
    """

    number_classes = len(set(np.array(shifted_labels)))
    one_hot_encoding = np.eye(number_classes)[shifted_labels.astype(int)]

    # one hot encoding with (-1 , 1)
    return 2 * one_hot_encoding - 1


def ssl_data_sample(y, sample_size):
    """
    :param y: array of labels for our dataset
    :param sample_size: number of samples required

    :return: permutation of our original indices such that the first
    """
    idxs = np.random.choice(
        list(np.where(y == -1.0)[0]), size=sample_size, replace=False
    ).tolist()
    idxs += np.random.choice(
        list(np.where(y == 1.0)[0]), size=sample_size, replace=False
    ).tolist()
    idxs += [i for i in range(y.shape[0]) if i not in idxs]
    return idxs


def knn_adjacency_matrix(x, k):
    """
    Given a design matrix x, generate an adjacency matrix of the K-NN graph of x
    """
    tree = KDTree(x)
    n = x.shape[0]
    graph = tree.query(x, k)[1][:, 1:]
    w = np.zeros((n, n))
    for i in range(n):
        w[i, graph[i, :]] = 1
        w[graph[i, :], i] = 1
    return w
