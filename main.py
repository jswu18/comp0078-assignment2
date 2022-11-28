import os

import jax
import numpy as np
import pandas as pd

from src.constants import DATA_FOLDER, DEFAULT_SEED, OUTPUTS_FOLDER
from src.solutions.q2 import q2
if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    # Question 1
    PART_1_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part1")
    if not os.path.exists(PART_1_OUTPUT_FOLDER):
        os.makedirs(PART_1_OUTPUT_FOLDER)

    # Question 2
    PART_2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "part2")
    if not os.path.exists(PART_2_OUTPUT_FOLDER):
        os.makedirs(PART_2_OUTPUT_FOLDER)

    q2()
