from dataclasses import dataclass
from typing import List, Tuple

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import vmap

from src.models.helpers import TrainTestData, make_folds
from src.models.kernels import BaseKernel
from src.models.multi_class import perceptron


@dataclass
class Performance:
    mean: np.ndarray
    stdev: np.ndarray

    def build_df(self, kernel_parameters, kernel_parameter_name, index):
        df = pd.DataFrame(
            data=[
                [
                    f"{'%.2f' % (100 * self.mean[i])}%±{'%.2f' % (100 * self.stdev[i])}%"
                    for i in range(len(kernel_parameters))
                ]
            ],
            columns=[
                f"{kernel_parameter_name}={kernel_parameters[i]}"
                for i in range(len(kernel_parameters))
            ],
        )
        df.index = [index]
        return df


def _analyse_prediction(actuals, predictions) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param actuals: (N_1,...,N_M, number_of_test_points, number_classes)
    :param predictions: (number_parameters, N_1,...,N_M, number_of_test_points, number_classes)
    :return: comparison of labels (number_parameters, N_1,...,N_M, number_of_test_points)
             and
             confusion matrix
    """
    number_of_parameters = predictions.shape[0]
    number_of_classes = predictions.shape[-1]
    actual_classes = np.argmax(actuals, axis=-1)
    prediction_classes = np.argmax(predictions, axis=-1)

    confusion_matrix = np.zeros(
        (number_of_parameters, number_of_classes, number_of_classes)
    )
    for i in range(number_of_parameters):
        np.add.at(
            confusion_matrix[i, ...], (actual_classes, prediction_classes[i, ...]), 1
        )
    return actual_classes[None, ...] == prediction_classes, confusion_matrix


def _train_and_test(
    data: TrainTestData,
    kernel_class: BaseKernel,
    kernel_parameter_name,
    kernel_parameters,
    number_of_epochs,
):
    # train_gram = np.array([kernel_class(
    #         data.x_train, **{kernel_parameter_name: kernel_parameter}
    #     ) for kernel_parameter in kernel_parameters])
    # test_gram = np.array([kernel_class(
    #         data.x_train, data.x_test, **{kernel_parameter_name: kernel_parameter}
    #     ) for kernel_parameter in kernel_parameters])
    train_gram = vmap(
        lambda kernel_parameter: kernel_class(
            data.x_train, **{kernel_parameter_name: kernel_parameter}
        )
    )(kernel_parameters)
    test_gram = vmap(
        lambda kernel_parameter: kernel_class(
            data.x_train, data.x_test, **{kernel_parameter_name: kernel_parameter}
        )
    )(kernel_parameters)
    weights = perceptron.train(
        gram=train_gram,
        y=data.y_train,
        number_of_epochs=number_of_epochs,
    )

    train_predictions = perceptron.predict(weights, train_gram)
    test_predictions = perceptron.predict(weights, test_gram)
    train_comparisons, _ = _analyse_prediction(data.y_train, train_predictions)
    test_comparisons, test_confusion_matrix = _analyse_prediction(
        data.y_test, test_predictions
    )
    return (
        weights,
        np.mean(~train_comparisons, axis=-1),
        np.mean(~test_comparisons, axis=-1),
        test_confusion_matrix,
    )


def _q1(
    data: List[TrainTestData],
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    number_of_epochs,
):
    number_of_parameters = len(kernel_parameters)
    number_classes = data[0].y_train.shape[1]
    number_of_runs = len(data)
    train_errors = np.zeros((number_of_parameters, number_of_runs))
    test_errors = np.zeros((number_of_parameters, number_of_runs))
    test_confusion_matrix = np.zeros(
        (number_of_parameters, number_of_runs, number_classes, number_classes)
    )
    weights = []
    for (
        i,
        data_run,
    ) in enumerate(data):

        (
            w,
            train_errors[:, i],
            test_errors[:, i],
            test_confusion_matrix[:, i, :, :],
        ) = _train_and_test(
            data=data_run,
            kernel_class=kernel_class,
            kernel_parameter_name=kernel_parameter_name,
            kernel_parameters=kernel_parameters,
            number_of_epochs=number_of_epochs,
        )
        weights.append(w)
    return train_errors, test_errors, test_confusion_matrix, weights


def q1(
    data: List[TrainTestData],
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    number_of_epochs,
    df_performance_path,
):
    train_error, test_error, _, _ = _q1(
        data, kernel_class, kernel_parameters, kernel_parameter_name, number_of_epochs
    )
    train_performance = Performance(
        mean=np.mean(train_error, axis=-1),
        stdev=np.std(train_error, axis=-1),
    )
    test_performance = Performance(
        mean=np.mean(test_error, axis=-1),
        stdev=np.std(test_error, axis=-1),
    )

    df = pd.concat(
        [
            train_performance.build_df(
                kernel_parameters=[
                    "{:.1e}".format(float(x)) for x in kernel_parameters
                ],
                kernel_parameter_name=kernel_parameter_name,
                index="Train",
            ),
            test_performance.build_df(
                kernel_parameters=[
                    "{:.1e}".format(float(x)) for x in kernel_parameters
                ],
                kernel_parameter_name=kernel_parameter_name,
                index="Test",
            ),
        ]
    ).T
    df.to_csv(df_performance_path + ".csv")
    dfi.export(df, df_performance_path + ".png")


def q2(
    data: List[TrainTestData],
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    number_of_epochs,
    number_of_folds,
    labels,
    df_performance_path,
    df_confusion_matrix_path,
    most_difficult_images_path,
):
    number_of_parameters = len(kernel_parameters)
    number_of_runs = len(data)
    number_classes = data[0].y_train.shape[1]

    test_errors = np.zeros((number_of_parameters, number_of_runs, number_of_folds))
    for (
        i,
        data_run,
    ) in enumerate(data):
        folds = make_folds(
            x=data_run.x_train,
            y=data_run.y_train,
            number_of_folds=number_of_folds,
        )
        _, test_errors[:, i, :], _, _ = _q1(
            data=folds,
            kernel_class=kernel_class,
            kernel_parameters=kernel_parameters,
            kernel_parameter_name=kernel_parameter_name,
            number_of_epochs=number_of_epochs,
        )
    best_parameters_per_run = kernel_parameters[
        np.argmin(np.mean(test_errors, axis=-1), axis=0)
    ]
    test_errors = np.zeros(number_of_runs)
    test_confusion_matrix = np.zeros((number_of_runs, number_classes, number_classes))
    weights = []
    for i, best_parameter_per_run in enumerate(best_parameters_per_run):
        _, test_errors[i], test_confusion_matrix[i, :, :], w = _q1(
            data=[data[i]],
            kernel_class=kernel_class,
            kernel_parameters=best_parameter_per_run.reshape(
                1,
            ),
            kernel_parameter_name=kernel_parameter_name,
            number_of_epochs=number_of_epochs,
        )
        weights.append(w[0])
    test_performance = Performance(
        mean=np.mean(test_errors).reshape(
            -1,
        ),
        stdev=np.std(test_errors).reshape(
            -1,
        ),
    )
    df_test_performance = test_performance.build_df(
        kernel_parameters=[
            f"{'{:.1e}'.format(float(np.mean(best_parameters_per_run)))}±{'{:.1e}'.format(float(np.std(best_parameters_per_run)))}"
        ],
        kernel_parameter_name=f"Mean Optimal {kernel_parameter_name}",
        index="Test Error Rate",
    ).T
    df_test_performance.to_csv(df_performance_path + ".csv")
    dfi.export(df_test_performance, df_performance_path + ".png")

    test_confusion_matrix = (
        test_confusion_matrix / test_confusion_matrix.sum(axis=2)[..., None]
    )
    confusion_mean = np.round(100 * test_confusion_matrix.mean(axis=0), 2)
    np.fill_diagonal(confusion_mean, 0)
    confusion_stdev = np.round(100 * test_confusion_matrix.std(axis=0))
    np.fill_diagonal(confusion_stdev, 0)
    df = pd.DataFrame(
        [
            [
                f"{confusion_mean[i, j]}%±{confusion_stdev[i, j]}%"
                for j in range(number_classes)
            ]
            for i in range(number_classes)
        ],
        columns=labels,
    )
    df.index = labels
    df.to_csv(df_confusion_matrix_path + ".csv")
    dfi.export(df, df_confusion_matrix_path + ".png")

    # find hardest to predict images
    # for each model, predict on the entire dataset
    number_data_points = data[0].x.shape[0]
    predictions = np.zeros((number_of_runs, number_data_points, number_classes))
    for i, w in enumerate(weights):
        gram = kernel_class(
            data[i].x_train,
            data[0].x,
            **{kernel_parameter_name: best_parameters_per_run[i]},
        )
        predictions[i, :, :] = perceptron.predict(w, gram)
    comparisons, _ = _analyse_prediction(data[0].y, predictions)
    most_difficult_image_indices = np.argpartition(np.sum(~comparisons, axis=0), -5)[
        -5:
    ]
    most_difficult_images = data[0].x[most_difficult_image_indices]
    most_difficult_labels = labels[
        np.argmax(data[0].y[most_difficult_image_indices], axis=1)
    ]
    fig, ax = plt.subplots(1, len(most_difficult_image_indices))
    fig.set_size_inches(5, 1.5)
    for i in range(len(most_difficult_image_indices)):
        ax[i].imshow(most_difficult_images[i].reshape(16, 16))
        ax[i].title.set_text(f"Label={most_difficult_labels[i]}")
        ax[i].axis("off")
    plt.savefig(most_difficult_images_path)
    plt.close(fig)
