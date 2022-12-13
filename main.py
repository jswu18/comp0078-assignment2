import os

import jax
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.constants import (
    DEFAULT_SEED,
    OUTPUTS_FOLDER,
    PART_1_DATA_SET_PATH,
    PART_1_MINI_TEST_SET_PATH,
    PART_1_MINI_TRAIN_SET_PATH,
)
from src.models.helpers import TrainTestData, one_hot_encode, split_train_test_data
from src.models.kernels import GaussianKernel, PolynomialKernel
from src.models.single_class.linear_regression_classifier import (
    LinearRegressionClassifier,
)
from src.models.single_class.one_nn import OneNN
from src.models.single_class.perceptron import Perceptron
from src.models.single_class.winnow import Winnow
from src.solutions import part_1, part_2, part_3

np.random.seed(DEFAULT_SEED)


def part_1_preprocess(data):
    x_data, labels = data[:, 1:], data[:, 0]
    # shift so that our labels start at zero
    shifted_labels = labels - np.min(labels)
    y_data = one_hot_encode(shifted_labels)
    return x_data, y_data, labels


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")

    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    # Question 1
    PART_1_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part1")
    if not os.path.exists(PART_1_OUTPUT_FOLDER):
        os.makedirs(PART_1_OUTPUT_FOLDER)

    raw_data = np.genfromtxt(PART_1_DATA_SET_PATH)
    x, y, labels = part_1_preprocess(raw_data)

    raw_data_mini_train = np.genfromtxt(PART_1_MINI_TRAIN_SET_PATH)
    x_mini_train, y_mini_train, _ = part_1_preprocess(raw_data_mini_train)
    raw_data_mini_test = np.genfromtxt(PART_1_MINI_TEST_SET_PATH)
    x_mini_test, y_mini_test, _ = part_1_preprocess(raw_data_mini_test)
    x_mini = np.concatenate((x_mini_train, x_mini_test), axis=0)
    y_mini = np.concatenate((y_mini_train, y_mini_test), axis=0)

    data_mini = TrainTestData(
        x_train=x_mini_train,
        y_train=y_mini_train,
        x_test=x_mini_test,
        y_test=y_mini_test,
    )

    number_of_runs = 20
    percent_split = 0.8
    test_error_convergence_rate = 0

    # POLYNOMIAL KERNEL
    part_1.q1(
        data_mini=data_mini,
        data=[
            split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
        ],
        kernel_class=PolynomialKernel(),
        kernel_parameters=np.arange(1, 8),
        kernel_parameter_name="degree",
        test_error_convergence_rate=test_error_convergence_rate,
        df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q1"),
    )

    part_1.q2(
        data_mini=data_mini,
        data=[
            split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
        ],
        kernel_class=PolynomialKernel(),
        kernel_parameters=np.arange(1, 8),
        kernel_parameter_name="degree",
        test_error_convergence_rate=test_error_convergence_rate,
        number_of_folds=5,
        labels=np.sort(list(set(labels))).astype(int),
        df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q2"),
        df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q3_confusion"),
        most_difficult_images_path=os.path.join(PART_1_OUTPUT_FOLDER, "q4.png"),
    )

    # GAUSSIAN KERNEL
    # find best parameters
    # part_1.q1(
    #     data_mini=data_mini,
    #     data=[
    #         split_train_test_data(x_mini, y_mini, percent_split)
    #         for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.logspace(-5, 1, 20),
    #     kernel_parameter_name="sigma",
    #     test_error_convergence_rate=test_error_convergence_rate,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_1-mini"),
    # )
    # df_gaussian_kernel_performance = pd.read_csv(
    #     os.path.join(PART_1_OUTPUT_FOLDER, "q5_1-mini.csv")
    # )
    # best_parameter_idx = np.argmin(df_gaussian_kernel_performance["Test Error"])
    # best_gaussian_kernel_params = list(
    #     df_gaussian_kernel_performance.iloc[
    #         best_parameter_idx - 1 : best_parameter_idx + 2, 0
    #     ]
    # )
    # best_gaussian_kernel_params = [
    #     float(x.split("=")[1])
    #     for x in [best_gaussian_kernel_params[0], best_gaussian_kernel_params[-1]]
    # ]

    # # use best parameters
    # part_1.q1(
    #     data_mini=data_mini,
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.linspace(
    #         best_gaussian_kernel_params[0], best_gaussian_kernel_params[1], 7
    #     ),
    #     kernel_parameter_name="sigma",
    #     test_error_convergence_rate=test_error_convergence_rate,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_1"),
    # )
    # part_1.q2(
    #     data_mini=data_mini,
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.linspace(
    #         best_gaussian_kernel_params[0], best_gaussian_kernel_params[1], 7
    #     ),
    #     kernel_parameter_name="sigma",
    #     test_error_convergence_rate=test_error_convergence_rate,
    #     number_of_folds=5,
    #     labels=np.sort(list(set(labels))).astype(int),
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_2"),
    #     df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_confusion"),
    #     most_difficult_images_path=os.path.join(
    #         PART_1_OUTPUT_FOLDER, "q5_most_difficult.png"
    #     ),
    # )

    # # ALTERNATIVE MULTICLASS ALGORITHM
    # part_1.q1(
    #     data_mini=data_mini,
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=PolynomialKernel(),
    #     kernel_parameters=np.arange(1, 8),
    #     kernel_parameter_name="degree",
    #     test_error_convergence_rate=test_error_convergence_rate,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q6"),
    #     use_default_update_method=False,
    # )

    # part_1.q2(
    #     data_mini=data_mini,
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=PolynomialKernel(),
    #     kernel_parameters=np.arange(1, 8),
    #     kernel_parameter_name="degree",
    #     test_error_convergence_rate=test_error_convergence_rate,
    #     number_of_folds=5,
    #     labels=np.sort(list(set(labels))).astype(int),
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q6_performance"),
    #     df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q6_confusion"),
    #     most_difficult_images_path=os.path.join(
    #         PART_1_OUTPUT_FOLDER, "q6_most_difficult.png"
    #     ),
    #     use_default_update_method=False,
    # )

    # Question 2
    PART_2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part2")
    if not os.path.exists(PART_2_OUTPUT_FOLDER):
        os.makedirs(PART_2_OUTPUT_FOLDER)
    part_2.q2()

    # # Question 3
    # PART_3_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part3")
    # if not os.path.exists(PART_3_OUTPUT_FOLDER):
    #     os.makedirs(PART_3_OUTPUT_FOLDER)
    # LOAD_PREVIOUS_RESULTS = True

    # candidate_complexity_functions = {
    #     "linear": {
    #         "fit": lambda x, y: np.polyfit(x, y, 1),
    #         "f": lambda x, coeff: np.poly1d(coeff)(x),
    #     },
    #     "quadratic": {
    #         "fit": lambda x, y: np.polyfit(x, y, 2),
    #         "f": lambda x, coeff: np.poly1d(coeff)(x),
    #     },
    #     "logarithm": {
    #         "fit": lambda x, y: curve_fit(
    #             lambda t, a, b, c: np.abs(a) * np.log(b * t) + c, x, y
    #         )[0],
    #         "f": lambda x, coeff: np.abs(coeff[0]) * np.log(coeff[1] * x) + coeff[2],
    #     },
    #     "exponential": {
    #         "fit": lambda x, y: curve_fit(
    #             lambda t, a, b, c: np.abs(a) * np.exp(b * t) + c, x, y
    #         )[0],
    #         "f": lambda x, coeff: np.abs(coeff[0]) * np.exp(coeff[1] * x) + coeff[2],
    #     },
    #     # "2^n": {
    #     #     "fit": lambda x, y: curve_fit(
    #     #         lambda t, a, b, c: a * (2 ** (b * t)) + c, x, y
    #     #     )[0],
    #     #     "f": lambda x, coeff: coeff[0] * (2 ** (coeff[1] * x)) + coeff[2],
    #     # },
    # }
    # part_3.a(
    #     model=OneNN(),
    #     dimensions=np.arange(1, 21),
    #     number_of_trials=100,
    #     m_test=200,
    #     candidate_complexity_functions=candidate_complexity_functions,
    #     figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_one_nn"),
    #     load_previous_results=LOAD_PREVIOUS_RESULTS,
    # )
    # part_3.a(
    #     model=Winnow(),
    #     dimensions=np.arange(1, 101),
    #     number_of_trials=500,
    #     m_test=200,
    #     candidate_complexity_functions=candidate_complexity_functions,
    #     figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_winnow"),
    #     load_previous_results=LOAD_PREVIOUS_RESULTS,
    # )
    # part_3.a(
    #     model=LinearRegressionClassifier(),
    #     dimensions=np.arange(1, 101),
    #     number_of_trials=500,
    #     m_test=200,
    #     candidate_complexity_functions=candidate_complexity_functions,
    #     figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_lin_reg"),
    #     load_previous_results=LOAD_PREVIOUS_RESULTS,
    # )
    # part_3.a(
    #     model=Perceptron(kernel=PolynomialKernel(), kernel_kwargs={"degree": 1}),
    #     dimensions=np.arange(1, 51),
    #     number_of_trials=100,
    #     m_test=200,
    #     candidate_complexity_functions=candidate_complexity_functions,
    #     figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_perceptron"),
    #     load_previous_results=LOAD_PREVIOUS_RESULTS,
    # )
