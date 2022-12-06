import os

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.constants import (
    DATA_FOLDER,
    DEFAULT_SEED,
    OUTPUTS_FOLDER,
    PART_1_DATA_SET_PATH,
    PART_1_MINI_TRAIN_SET_PATH,
)
from src.models.helpers import one_hot_encode, split_train_test_data
from src.models.kernels import GaussianKernel, PolynomialKernel
from src.models.linear_regression_classifier import LinearRegressionClassifier
from src.models.one_nn import OneNN
from src.models.single_class_perceptron import SingleClassPerceptron
from src.models.winnow import Winnow
from src.solutions import part_1, part_3
from src.solutions.part_2 import q2

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    # # Question 1
    # PART_1_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part1")
    # if not os.path.exists(PART_1_OUTPUT_FOLDER):
    #     os.makedirs(PART_1_OUTPUT_FOLDER)
    #
    # data = np.genfromtxt(PART_1_DATA_SET_PATH)
    # x, labels = data[:, 1:], data[:, 0]
    # # shift so that our labels start at zero
    # shifted_labels = labels - jnp.min(labels)
    # y = one_hot_encode(shifted_labels)
    #
    # number_of_runs = 20
    # percent_split = 0.8
    #
    # part_1.q1(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=PolynomialKernel(),
    #     kernel_parameters=np.arange(1, 8),
    #     kernel_parameter_name="degree",
    #     number_of_epochs=2,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q1"),
    # )
    #
    # part_1.q2(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=PolynomialKernel(),
    #     kernel_parameters=np.arange(1, 8),
    #     kernel_parameter_name="degree",
    #     number_of_epochs=2,
    #     number_of_folds=5,
    #     labels=np.sort(list(set(labels))).astype(int),
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q2"),
    #     df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q3_confusion"),
    #     most_difficult_images_path=os.path.join(PART_1_OUTPUT_FOLDER, "q4.png"),
    # )
    #
    # part_1.q1(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.arange(2e-3, 3e-2, 4e-3),
    #     kernel_parameter_name="sigma",
    #     number_of_epochs=2,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_1"),
    # )
    #
    # part_1.q2(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.arange(2e-3, 3e-2, 4e-3),
    #     kernel_parameter_name="sigma",
    #     number_of_epochs=2,
    #     number_of_folds=5,
    #     labels=np.sort(list(set(labels))).astype(int),
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_2"),
    #     df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_confusion"),
    #     most_difficult_images_path=os.path.join(
    #         PART_1_OUTPUT_FOLDER, "q5_most_difficult.png"
    #     ),
    # )
    #
    # # Question 2
    # PART_2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part2")
    # if not os.path.exists(PART_2_OUTPUT_FOLDER):
    #     os.makedirs(PART_2_OUTPUT_FOLDER)
    # q2()

    # Question 3
    PART_3_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part3")
    if not os.path.exists(PART_3_OUTPUT_FOLDER):
        os.makedirs(PART_3_OUTPUT_FOLDER)
    part_3.a(
        model=OneNN(),
        dimensions=np.arange(1, 11),
        number_train_points=np.arange(1, 101),
        number_of_trials=10,
        m_test=20,
        figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_one_nn.png"),
    )
    part_3.a(
        model=Winnow(),
        dimensions=np.arange(2, 11),
        number_train_points=np.arange(2, 101),
        number_of_trials=10,
        m_test=20,
        figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_winnow.png"),
    )
    part_3.a(
        model=LinearRegressionClassifier(),
        dimensions=np.arange(2, 11),
        number_train_points=np.arange(2, 101),
        number_of_trials=10,
        m_test=20,
        figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_lin_reg.png"),
    )
    part_3.a(
        model=SingleClassPerceptron(
            kernel=PolynomialKernel(), kernel_kwargs={"degree": 1}
        ),
        dimensions=np.arange(3, 11),
        number_train_points=np.arange(2, 50),
        number_of_trials=10,
        m_test=20,
        figure_save_path=os.path.join(PART_3_OUTPUT_FOLDER, "q1a_perceptron.png"),
    )
