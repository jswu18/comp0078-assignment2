import matplotlib.pyplot as plt
import numpy as np

from src.models import linear_regression_classifier
from src.models.one_nn import one_nn_predict


def generate_data(m, n):
    x = np.random.choice([-1, 1], size=(m, n))
    y = x[:, :1]
    return x, y


dimensions = np.arange(1, 21)
number_train_points = np.arange(1, 50)
number_of_iterations = 1000
percentage = 1e-2
m_test = int(1e3)

mean_number_train_points = np.zeros((len(dimensions)))
squared_mean_number_train_points = np.zeros((len(dimensions)))

for i, n in enumerate(dimensions):
    for j in range(number_of_iterations):
        for k, m in enumerate(number_train_points):
            x_train, y_train = generate_data(m, n)
            w = linear_regression_classifier.train(x_train, y_train)

            x_test, y_test = generate_data(m_test, n)
            y_pred = linear_regression_classifier.predict(w, x_test)
            if np.mean(y_pred != y_test) < 0.1:
                mean_number_train_points[i] = (mean_number_train_points[i] * j + m) / (
                    j + 1
                )
                squared_mean_number_train_points[i] = (
                    squared_mean_number_train_points[i] * j + m**2
                ) / (j + 1)
                break

variance_number_train_points = (
    squared_mean_number_train_points - mean_number_train_points**2
)

plt.errorbar(
    dimensions, mean_number_train_points, yerr=variance_number_train_points, capsize=2
)


dimensions = np.arange(1, 11)
number_train_points = np.arange(1, 201)
number_of_iterations = 1000
percentage = 1e-2
m_test = int(1e3)

mean_number_train_points = np.ones((len(dimensions))) * np.nan
squared_mean_number_train_points = np.ones((len(dimensions))) * np.nan

for i, n in enumerate(dimensions):
    for _ in range(number_of_iterations):
        j = 0
        for k, m in enumerate(number_train_points):
            x_train, y_train = generate_data(m, n)
            x_test, y_test = generate_data(m_test, n)

            y_pred = one_nn_predict(x_train, y_train, x_test)
            if np.mean(y_pred != y_test) < 0.1:
                mean_number_train_points[i] = np.nansum(
                    [mean_number_train_points[i] * j, m]
                ) / (j + 1)
                squared_mean_number_train_points[i] = np.nansum(
                    [squared_mean_number_train_points[i] * j + m**2]
                ) / (j + 1)
                j += 1
                break

variance_number_train_points = (
    squared_mean_number_train_points - mean_number_train_points**2
)
plt.errorbar(
    dimensions, mean_number_train_points, yerr=variance_number_train_points, capsize=2
)
