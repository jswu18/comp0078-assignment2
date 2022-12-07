import numpy as np

from src.models.single_class.model import Model


class Winnow(Model):
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
                  (number_of_dimensions)
        :param x: input data point (across all experiments)
                  (number_points, number_of_dimensions)
        :return: winnow prediction
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
                  (number_of_dimensions)
        :param x: input data point (across all experiments)
                     (number_of_dimensions)
        :param y: response
        :return: new weight update (number_of_dimensions)
        """
        prediction = Winnow._predict(w, x)  # (N_1,...,N_M)
        return np.multiply(
            w, 2 ** np.multiply((y - prediction)[..., None], x).astype(float)
        )

    def fit_predict(self, x_train, y_train, x_test, **kwargs):
        """
        Vectorised winnow training by training multiple trials, etc. in parallel
        The winnow is trained one data point at a time, the new weights depend on the weights of the previous
        step, however each trial is independent, thus for our experiments, we can perform this in parallel.

        All input matrices will share the first M dimensions representing the different independent experiments
        that we want to train for. N_i will be the size of the ith dimension, and i = 1, 2, ..., M

        :param x: design matrix
                  (number_training_points, number_of_dimensions)
        :param y: matrix of responses, the response for all parameter trials will be the same
                  (number_training_points)
        :param **kwargs: number_of_epochs: number of epochs to train model
        :return:
        """
        if "number_of_epochs" in kwargs:
            number_of_epochs = kwargs["number_of_epochs"]
        else:
            number_of_epochs = 1

        w = np.ones(x_train.shape[1])
        y_train = self._preprocess(y_train)
        for _ in range(number_of_epochs):
            for i in range(1, x_train.shape[0]):
                w = self._compute_update(w, y=y_train[i], x=x_train[i, :])
        return self._postprocess(self._predict(w, x_test))
