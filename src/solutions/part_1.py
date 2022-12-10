from dataclasses import dataclass
from typing import List, Tuple

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def _build_confusion(confusion_matrix, x_idx, y_idx):
    if confusion_matrix.ndim > 2:
        for i in range(confusion_matrix.shape[0]):
            confusion_matrix[i, ...] = _build_confusion(
                confusion_matrix[i, ...],
                x_idx[i, ...],
                y_idx[i, ...],
            )
    else:
        np.add.at(confusion_matrix, (x_idx, y_idx), 1)
    return confusion_matrix


def _analyse_prediction(actuals, predictions) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param actuals: (N_1,...,N_M, number_of_test_points, number_classes)
    :param predictions: (N_1,...,N_M, number_of_test_points, number_classes)
    :return: comparison of labels (N_1,...,N_M, number_of_test_points)
             and
             confusion matrix (N_1,...,N_M, number_classes, number_classes)
    """
    number_of_classes = predictions.shape[-1]
    actual_classes = np.argmax(actuals, axis=-1)
    prediction_classes = np.argmax(predictions, axis=-1)

    confusion_matrix = np.zeros(
        list(actuals.shape[:-2]) + [number_of_classes, number_of_classes]
    )
    confusion_matrix = _build_confusion(
        confusion_matrix, actual_classes, prediction_classes
    )
    return actual_classes == prediction_classes, confusion_matrix


def _train_and_test(
    train_gram,
    test_gram,
    y_train,
    y_test,
    number_of_epochs,
    use_default_update_method: bool = True,
):
    weights = perceptron.train(
        w=np.zeros(y_train.shape),
        gram=train_gram,
        y=y_train,
        number_of_epochs=number_of_epochs,
        use_default_update_method=use_default_update_method,
    )

    train_predictions = perceptron.predict(weights, train_gram)
    train_comparisons, _ = _analyse_prediction(y_train, train_predictions)

    test_predictions = perceptron.predict(weights, test_gram)
    test_comparisons, test_confusion_matrix = _analyse_prediction(
        y_test, test_predictions
    )

    return (
        weights,
        np.mean(~train_comparisons, axis=-1),
        np.mean(~test_comparisons, axis=-1),
        test_confusion_matrix,
    )


def _approximate_epochs(
    data,
    kernel_class: BaseKernel,
    kernel_parameter_name: str,
    kernel_parameter: float,
    test_error_convergence_rate,
    use_default_update_method: bool = True,
):
    train_pre_gram = kernel_class.pre_gram(data.x_train)
    test_pre_gram = kernel_class.pre_gram(data.x_train, data.x_test)
    train_gram = kernel_class.post_gram(
        train_pre_gram,
        **{kernel_parameter_name: kernel_parameter},
    )
    test_gram = kernel_class.post_gram(
        test_pre_gram,
        **{kernel_parameter_name: kernel_parameter},
    )

    weights = np.zeros(data.y_train.shape)
    previous_error = 1
    epochs = 0
    while True:
        weights = perceptron.train(
            w=weights,
            gram=train_gram,
            y=data.y_train,
            number_of_epochs=1,
            use_default_update_method=use_default_update_method,
        )
        test_predictions = perceptron.predict(weights, test_gram)
        test_comparisons, test_confusion_matrix = _analyse_prediction(
            data.y_test, test_predictions
        )
        current_error = np.mean(~test_comparisons, axis=-1)
        if np.abs(previous_error - current_error) > test_error_convergence_rate:
            previous_error = current_error
            epochs += 1
        else:
            break
    return epochs


def _q1(
    data_run: TrainTestData,
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    number_of_epochs,
    use_default_update_method: bool = True,
):
    number_classes = data_run.y_train.shape[1]
    number_of_parameters = len(kernel_parameters)
    train_errors = np.zeros(number_of_parameters)
    test_errors = np.zeros(number_of_parameters)
    test_confusion_matrix = np.zeros(
        (number_of_parameters, number_classes, number_classes)
    )
    weights = []
    train_pre_gram = kernel_class.pre_gram(data_run.x_train)
    test_pre_gram = kernel_class.pre_gram(data_run.x_train, data_run.x_test)
    for i, kernel_parameter in enumerate(kernel_parameters):
        (
            w,
            train_errors[i],
            test_errors[i],
            test_confusion_matrix[i, :, :],
        ) = _train_and_test(
            train_gram=kernel_class.post_gram(
                train_pre_gram, **{kernel_parameter_name: kernel_parameter}
            ),
            test_gram=kernel_class.post_gram(
                test_pre_gram, **{kernel_parameter_name: kernel_parameter}
            ),
            y_train=data_run.y_train,
            y_test=data_run.y_test,
            number_of_epochs=number_of_epochs[i],
            use_default_update_method=use_default_update_method,
        )
        weights.append(w)
    return train_errors, test_errors, test_confusion_matrix, weights


def q1(
    data_mini: TrainTestData,
    data: List[TrainTestData],
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    test_error_convergence_rate,
    df_performance_path,
    use_default_update_method: bool = True,
):
    number_of_runs = len(data)
    number_kernel_parameters = len(kernel_parameters)
    train_errors = np.zeros((number_kernel_parameters, number_of_runs))
    test_errors = np.zeros((number_kernel_parameters, number_of_runs))

    # calculate number of epochs to train
    number_of_epochs = np.zeros(number_kernel_parameters)
    for i, kernel_parameter in enumerate(kernel_parameters):
        number_of_epochs[i] = _approximate_epochs(
            data=data_mini,
            kernel_class=kernel_class,
            kernel_parameter_name=kernel_parameter_name,
            kernel_parameter=kernel_parameter,
            test_error_convergence_rate=test_error_convergence_rate,
            use_default_update_method=use_default_update_method,
        )
    df_epochs = pd.DataFrame(
        number_of_epochs.reshape(1, -1).astype(int),
        columns=[
            f"{kernel_parameter_name}=" + "{:.1e}".format(float(x))
            for x in kernel_parameters
        ],
    )
    df_epochs.index = ["Number of Training Epochs"]

    for (
        i,
        data_run,
    ) in tqdm(enumerate(data)):
        train_errors[:, i], test_errors[:, i], _, _ = _q1(
            data_run,
            kernel_class,
            kernel_parameters,
            kernel_parameter_name,
            number_of_epochs,
            use_default_update_method,
        )
    # for i, kernel_parameter in tqdm(enumerate(kernel_parameters)):
    #     train_errors[i, :], test_errors[i, :], _, _ = _q1(
    #         data, kernel_class, kernel_parameter, kernel_parameter_name, number_of_epochs, use_default_update_method
    #     )
    train_performance = Performance(
        mean=np.mean(train_errors, axis=-1),
        stdev=np.std(train_errors, axis=-1),
    )
    test_performance = Performance(
        mean=np.mean(test_errors, axis=-1),
        stdev=np.std(test_errors, axis=-1),
    )

    df = pd.concat(
        [
            df_epochs,
            train_performance.build_df(
                kernel_parameters=[
                    "{:.1e}".format(float(x)) for x in kernel_parameters
                ],
                kernel_parameter_name=kernel_parameter_name,
                index="Train Error",
            ),
            test_performance.build_df(
                kernel_parameters=[
                    "{:.1e}".format(float(x)) for x in kernel_parameters
                ],
                kernel_parameter_name=kernel_parameter_name,
                index="Test Error",
            ),
        ]
    ).T
    df.to_csv(df_performance_path + ".csv")
    dfi.export(df, df_performance_path + ".png")


def q2(
    data_mini: TrainTestData,
    data: List[TrainTestData],
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    test_error_convergence_rate,
    number_of_folds,
    labels,
    df_performance_path,
    df_confusion_matrix_path,
    most_difficult_images_path,
    use_default_update_method: bool = True,
):
    number_of_parameters = len(kernel_parameters)
    number_of_runs = len(data)
    number_classes = data[0].y_train.shape[1]

    # calculate number of epochs to train
    number_of_epochs = np.zeros(number_of_parameters)
    for i, kernel_parameter in enumerate(kernel_parameters):
        number_of_epochs[i] = _approximate_epochs(
            data=data_mini,
            kernel_class=kernel_class,
            kernel_parameter_name=kernel_parameter_name,
            kernel_parameter=kernel_parameter,
            test_error_convergence_rate=test_error_convergence_rate,
            use_default_update_method=use_default_update_method,
        )
    df_epochs = pd.DataFrame(
        number_of_epochs.reshape(1, -1).astype(int),
        columns=[
            f"{kernel_parameter_name}=" + "{:.1e}".format(float(x))
            for x in kernel_parameters
        ],
    )
    df_epochs.index = ["Number of Training Epochs"]

    test_errors = np.zeros((number_of_parameters, number_of_runs, number_of_folds))
    for (
        i,
        data_run,
    ) in tqdm(enumerate(data)):
        folds = make_folds(
            x=data_run.x_train,
            y=data_run.y_train,
            number_of_folds=number_of_folds,
        )
        for j, fold in enumerate(folds):
            _, test_errors[:, i, j], _, _ = _q1(
                fold,
                kernel_class,
                kernel_parameters,
                kernel_parameter_name,
                number_of_epochs,
                use_default_update_method,
            )
    best_parameters_per_run = kernel_parameters[
        np.argmin(np.mean(test_errors, axis=-1), axis=0)
    ]
    best_parameters_number_of_epochs = number_of_epochs[
        np.argmin(np.mean(test_errors, axis=-1), axis=0)
    ]
    train_errors = np.zeros(number_of_runs)
    test_errors = np.zeros(number_of_runs)
    test_confusion_matrix = np.zeros((number_of_runs, number_classes, number_classes))
    weights = []
    for i, best_parameter_per_run in tqdm(enumerate(best_parameters_per_run)):
        train_errors[i], test_errors[i], test_confusion_matrix[i, :, :], w = _q1(
            data[i],
            kernel_class,
            [best_parameter_per_run],
            kernel_parameter_name,
            best_parameters_number_of_epochs,
            use_default_update_method,
        )
        weights.append(w[0])

    df_optimal_performance_runs = pd.DataFrame(
        data=np.array(
            [
                [f"{int(x)}" for x in np.arange(1, number_of_runs + 1)],
                ["{:.1e}".format(float(x)) for x in best_parameters_per_run],
                [f"{int(x)}" for x in best_parameters_number_of_epochs],
                [f"{'%.2f' % (100 * x)}%" for x in train_errors],
                [f"{'%.2f' % (100 * x)}%" for x in test_errors],
            ]
        ).T,
        columns=[
            "Run",
            f"Optimal {kernel_parameter_name}",
            "Number of Training Epochs",
            "Train Error",
            "Test Error",
        ],
    )

    df_mean_optimal_performance = pd.DataFrame(
        data=np.array(
            [
                ["Across Runs"],
                [
                    f"{'{:.1e}'.format(float(np.mean(best_parameters_per_run)))}±{'{:.1e}'.format(float(np.std(best_parameters_per_run)))}"
                ],
                [
                    f"{'%.2f' % np.mean(best_parameters_number_of_epochs)}±{'%.2f' % np.std(best_parameters_number_of_epochs)}"
                ],
                [
                    f"{'%.2f' % (100 * np.mean(train_errors))}%±{'%.2f' % (100 * np.std(train_errors))}%"
                ],
                [
                    f"{'%.2f' % (100 * np.mean(test_errors))}%±{'%.2f' % (100 * np.std(test_errors))}%"
                ],
            ]
        ).T,
        columns=[
            "Run",
            f"Optimal {kernel_parameter_name}",
            "Number of Training Epochs",
            "Train Error",
            "Test Error",
        ],
    )
    df_optimal_performance = pd.concat(
        [df_optimal_performance_runs, df_mean_optimal_performance]
    )
    # test_performance = Performance(
    #     mean=np.mean(test_errors).reshape(
    #         -1,
    #     ),
    #     stdev=np.std(test_errors).reshape(
    #         -1,
    #     ),
    # )
    # df_test_performance = test_performance.build_df(
    #     kernel_parameters=[
    #         f"{'{:.1e}'.format(float(np.mean(best_parameters_per_run)))}±{'{:.1e}'.format(float(np.std(best_parameters_per_run)))}"
    #     ],
    #     kernel_parameter_name=f"mean optimal {kernel_parameter_name}",
    #     index="Test Error Rate",
    # ).T
    df_optimal_performance = df_optimal_performance.set_index("Run")
    df_optimal_performance.to_csv(df_performance_path + ".csv")
    dfi.export(df_optimal_performance, df_performance_path + ".png")

    test_confusion_matrix = (
        test_confusion_matrix / test_confusion_matrix.sum(axis=2)[..., None]
    )
    confusion_mean = test_confusion_matrix.mean(axis=0)
    np.fill_diagonal(confusion_mean, 0)
    plt.figure()
    plt.imshow(confusion_mean)
    plt.colorbar()
    plt.xticks(
        np.arange(len(labels)),
        labels=labels,
    )
    plt.xlabel("Predicted Labels")
    plt.yticks(
        np.arange(len(labels)),
        labels=labels,
    )
    plt.ylabel("Actual Labels")
    plt.title(f"Test Set Error Confusion Matrix Mean (error range: [0, 1])")
    plt.savefig(df_confusion_matrix_path + "-imshow_mean.png")

    confusion_stdev = test_confusion_matrix.std(axis=0)
    np.fill_diagonal(confusion_stdev, 0)
    plt.figure()
    plt.imshow(confusion_stdev)
    plt.colorbar()
    plt.xticks(
        np.arange(len(labels)),
        labels=labels,
    )
    plt.xlabel("Predicted Labels")
    plt.yticks(
        np.arange(len(labels)),
        labels=labels,
    )
    plt.ylabel("Actual Labels")
    plt.title(f"Test Set Error Confusion Matrix St. Dev. (error range: [0, 1])")
    plt.savefig(df_confusion_matrix_path + "-imshow_stdev.png")

    confusion_mean = np.round(100 * confusion_mean, 2)
    confusion_stdev = np.round(100 * confusion_stdev, 2)
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
    df = pd.concat(
        [pd.concat([df], keys=["Predicted Labels"], axis=1)],
        keys=["Actual Labels"],
    )
    df.to_csv(df_confusion_matrix_path + ".csv")
    dfi.export(df, df_confusion_matrix_path + ".png")

    # find hardest to predict images
    # for each model, predict on the entire dataset
    number_data_points = data[0].x.shape[0]
    predictions = np.zeros((number_of_runs, number_data_points, number_classes))
    for i, w in enumerate(weights):
        pre_gram = kernel_class.pre_gram(
            data[i].x_train,
            data[0].x,
        )
        gram = kernel_class.post_gram(
            pre_gram,
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
    fig.set_size_inches(5, 2)
    for i in range(len(most_difficult_image_indices)):
        ax[i].imshow(most_difficult_images[i].reshape(16, 16))
        ax[i].title.set_text(f"Label={most_difficult_labels[i]}")
        ax[i].axis("off")
    plt.suptitle("Hardest to Predict Images")
    plt.savefig(most_difficult_images_path)
    plt.close(fig)
