import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.constants import (
    DATA_FOLDER,
    DEFAULT_SEED,
    OUTPUTS_FOLDER,
    PART_1_MINI_TRAIN_SET_PATH,
    PART_1_DATA_SET_PATH,
)
from src.solutions.q2 import q2
from src.solutions import part_1
from src.models.helpers import one_hot_encode, split_train_test_data
from src.models.kernels import PolynomialKernel, GaussianKernel
import jax

if __name__ == "__main__":
    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_enable_x64", True)

    # if not os.path.exists(OUTPUTS_FOLDER):
    #     os.makedirs(OUTPUTS_FOLDER)

    # # Question 1
    # PART_1_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part1")
    # if not os.path.exists(PART_1_OUTPUT_FOLDER):
    #     os.makedirs(PART_1_OUTPUT_FOLDER)

    # data = np.genfromtxt(PART_1_DATA_SET_PATH)
    # x, labels = data[:, 1:], data[:, 0]
    # # shift so that our labels start at zero
    # shifted_labels = labels - jnp.min(labels)
    # y = one_hot_encode(shifted_labels)

    # number_of_runs = 20
    # percent_split = 0.8

    # part_1.q1(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=PolynomialKernel(),
    #     kernel_parameters=np.arange(1, 8),
    #     kernel_parameter_name="degree",
    #     number_of_epochs=1,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q1.csv"),
    # )

    # part_1.q2(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=PolynomialKernel(),
    #     kernel_parameters=np.arange(1, 8),
    #     kernel_parameter_name="degree",
    #     number_of_epochs=1,
    #     number_of_folds=5,
    #     labels=np.sort(list(set(labels))).astype(int),
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q2.csv"),
    #     df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q2_confusion.csv"),
    #     most_difficult_images_path=os.path.join(PART_1_OUTPUT_FOLDER, "q3.png"),
    # )

    # part_1.q1(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.arange(2e-3, 3e-2, 4e-3),
    #     kernel_parameter_name="sigma",
    #     number_of_epochs=1,
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_1.csv"),
    # )

    # part_1.q2(
    #     data=[
    #         split_train_test_data(x, y, percent_split) for _ in range(number_of_runs)
    #     ],
    #     kernel_class=GaussianKernel(),
    #     kernel_parameters=np.arange(2e-3, 3e-2, 4e-3),
    #     kernel_parameter_name="sigma",
    #     number_of_epochs=1,
    #     number_of_folds=5,
    #     labels=np.sort(list(set(labels))).astype(int),
    #     df_performance_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_2.csv"),
    #     df_confusion_matrix_path=os.path.join(PART_1_OUTPUT_FOLDER, "q5_confusion.csv"),
    #     most_difficult_images_path=os.path.join(
    #         PART_1_OUTPUT_FOLDER, "q5_most_difficult.png"
    #     ),
    # )

    # # Question 2
    PART_2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part2")
    if not os.path.exists(PART_2_OUTPUT_FOLDER):
        os.makedirs(PART_2_OUTPUT_FOLDER)
    q2()
