import matplotlib.pyplot as plt
import numpy as np

from src.models.model import Model


def generate_data(m: int, n: int):
    """
    Generate data with the provided protocol
    :param m: number of examples
    :param n: number of dimensions
    :return: Tuple containing an array of patterns (m, n) and an array of labels (m,)
    """
    x = np.random.choice([-1, 1], size=(m, n))
    y = x[:, 0]
    return x, y


def a(
    model: Model,
    dimensions: np.ndarray,
    number_train_points: np.ndarray,
    number_of_trials: int,
    m_test: int,
    figure_save_path: str,
    generalisation_error: float = 0.1,
):
    number_of_dimensions = len(dimensions)
    train_points_for_generalisation_error = (
        np.ones((number_of_dimensions, number_of_trials)) * np.nan
    )

    for i, n in enumerate(dimensions):
        for j in range(number_of_trials):
            for k, m in enumerate(number_train_points):
                x_train, y_train = generate_data(m, n)
                model.fit(x_train, y_train)

                x_test, y_test = generate_data(m_test, n)
                y_predicted = model.predict(x_test)
                error_rate = np.mean(y_predicted != y_test)
                if error_rate < generalisation_error:
                    # update mean and squared mean vectors with nansum to ensure that the resulting vector indicates
                    # when the generalisation error requirement is reached
                    train_points_for_generalisation_error[i, j] = m
                    break
    plt.errorbar(
        dimensions,
        np.nanmean(train_points_for_generalisation_error, axis=1),
        yerr=np.nanstd(train_points_for_generalisation_error, axis=1),
        capsize=2,
    )
    plt.savefig(figure_save_path)
    plt.close()
