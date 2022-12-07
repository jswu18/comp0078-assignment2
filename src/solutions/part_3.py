from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.models.single_class.model import Model


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
    number_of_trials: int,
    m_test: int,
    figure_save_path: str,
    candidate_complexity_functions: Dict[str, Callable],
    generalisation_error: float = 0.1,
):
    number_of_dimensions = len(dimensions)

    # initialise as maximum training points such that in the event we do not find
    # an error rate below the generalisation error, we just take the maximum number
    # of training points, this will be a lower bound on the required number of training
    # points and our estimate will be biased slightly lower than the actual.
    train_points_for_generalisation_error = (
        np.ones((number_of_dimensions, number_of_trials)) * np.nan
    )

    for i, n in tqdm(enumerate(dimensions)):
        for j in tqdm(range(number_of_trials)):
            x_test, y_test = generate_data(m_test, n)
            error_rate = 1
            m = 0
            while error_rate > generalisation_error:
                m += 1
                x_train, y_train = generate_data(m, n)
                y_predicted = model.fit_predict(x_train, y_train, x_test)
                assert y_predicted.shape == y_test.shape
                error_rate = np.mean(y_predicted != y_test)
            train_points_for_generalisation_error[i, j] = m
    plt.errorbar(
        dimensions,
        np.mean(train_points_for_generalisation_error, axis=1),
        yerr=np.std(train_points_for_generalisation_error, axis=1),
        capsize=2,
    )
    plt.title(type(model).__name__)
    plt.xlabel("n")
    plt.ylabel("m")
    plt.savefig(figure_save_path + "_sample_complexity.png")
    plt.close()

    fig, ax = plt.subplots(1, len(candidate_complexity_functions))
    fig.set_size_inches(len(candidate_complexity_functions) * 5, 6)
    for i, function_name in enumerate(candidate_complexity_functions.keys()):
        f = candidate_complexity_functions[function_name]
        ax[i].plot(
            dimensions,
            np.divide(
                f(dimensions), np.mean(train_points_for_generalisation_error, axis=1)
            ),
        )
        ax[i].title.set_text(f"{function_name} Comparison")
    plt.suptitle(f"{type(model).__name__}: Complexity Function Comparison")
    plt.savefig(figure_save_path + "_complexity_function_comparison.png")
    plt.close()
