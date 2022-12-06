import numpy as np

from src.models.model import Model


class Winnow(Model):
    def __init__(self, w=None):
        self.w: np.ndarray = w
        super().__init__()

    @staticmethod
    def _preprocess(y):
        y[y == -1] = 0
        return y

    @staticmethod
    def _postprocess(y):
        y[y == 0] = -1
        return y

    @staticmethod
    def _predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        prediction for a single data point (across all experiments)
        :param w: weight matrix
                  (N_1,...,N_M, number_features)
        :param x: input data point (across all experiments)
                  (N_1,...,N_M, number_features)
        :return: winnow prediction
                 (N_1,...,N_M)
        """
        return (
            (np.mean(np.multiply(w, x), axis=-1) - 1)
            .clip(0, 1)
            .astype(bool)
            .astype(int)
        )

    @staticmethod
    def _compute_update(w: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        compute weight update for a single data point (across all experiments)
        :param w: weight matrix
                  (N_1,...,N_M, number_features)
        :param x: input data point (across all experiments)
                     (N_1,...,N_M, number_features)
        :param y: matrix of a responses (across all experiments)
                  (N_1,...,N_M)
        :return: new weight matrix (N_1,...,N_M, number_features)
        """
        prediction = Winnow._predict(w, x)  # (N_1,...,N_M)
        return np.multiply(
            w, 2 ** np.multiply((y - prediction)[..., None], x).astype(float)
        )

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Vectorised winnow training by training multiple trials, etc. in parallel
        The winnow is trained one data point at a time, the new weights depend on the weights of the previous
        step, however each trial is independent, thus for our experiments, we can perform this in parallel.

        All input matrices will share the first M dimensions representing the different independent experiments
        that we want to train for. N_i will be the size of the ith dimension, and i = 1, 2, ..., M

        :param x: design matrix
                  (number_features, number_training_points)
        :param y: matrix of responses, the response for all parameter trials will be the same
                  (N_1,...,N_M, number_training_points)
        :param **kwargs: number_of_epochs: number of epochs to train model
        :return:
        """
        if "number_of_epochs" in kwargs:
            number_of_epochs = kwargs["number_of_epochs"]
        else:
            number_of_epochs = 1

        self.w = np.ones((x.shape[:-1]))
        y = self._preprocess(y)
        for _ in range(number_of_epochs):
            for i in range(1, x.shape[-1]):
                self.w = self._compute_update(self.w, y=y[..., i], x=x[..., i])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        winnow prediction
        :param x: design matrix
                  (N_1,...,N_M, number_features, number_of_test_points)
        :return: predictions for test points
                 (N_1,...,N_M, number_of_test_points)
        """
        return self._postprocess(
            np.stack(
                [self._predict(self.w, x[..., i]) for i in range(x.shape[-1])], axis=-1
            )
        )
