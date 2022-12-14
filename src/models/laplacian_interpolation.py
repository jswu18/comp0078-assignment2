import numpy as np


class LaplacianInterpolation:
    @staticmethod
    def predict(w, y):
        """
        Note: we have ordered our graph nodes so the first l datapoints are the
        labelled datapoints.

        The following expression was given in the paper provided, and hence used in
        this example for a closed-form solution.

        :param w: 3-NN adjacency matrix for the datapoints in X
        :param y: set of labels for x_1, ..., x_l
        :return: y_hat: predicted labels for y_{l+1}, ... , y_{n}
        """
        s = y.shape[0]
        diagonal = np.diag(w.sum(0))
        return np.sign(np.linalg.pinv(diagonal[s:, s:] - w[s:, s:]) @ (w[s:, :s] @ y))


class LaplacianKernelInterpolation:
    @staticmethod
    def predict(w, y):
        """

        :param w: 3-NN adjacency matrix for the datapoints in X
        :param y: set of labels for x_1, ..., x_l
        :return: y_hat: predicted labels for y_{l+1}, ... , y_{n}
        """
        l_matrix = np.diag(w.sum(0)) - w
        m = y.shape[0]
        l_matrix_pinv = np.linalg.pinv(l_matrix)
        alpha = np.linalg.pinv(l_matrix_pinv[:m, :m]) @ y
        return np.sign(l_matrix_pinv[:m, :].T @ alpha)[m:]
