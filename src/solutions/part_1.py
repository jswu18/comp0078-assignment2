from dataclasses import dataclass
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import vmap

from src.models.helpers import TrainTestData, make_folds
from src.models.perceptron import Perceptron
from src.models.kernels import BaseKernel

def _convert_to_scientific_notation(x: float) -> str:
    """
    Convert value to string in scientific notation
    :param x: value to convert
    :return: string of x in scientific notation
    """
    return "{:.2e}".format(float(x))


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


def _build_folds_data(x, y, number_of_folds):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    number_of_runs = x.shape[0]
    for i in range(number_of_runs):
        folds = make_folds(
            x=x[i],
            y=y[i],
            number_of_folds=number_of_folds,
        )
        min_fold_size_train = np.min([fold.x_train.shape[0] for fold in folds])
        x_train.append(np.vstack([fold.x_train[None, :min_fold_size_train, ...] for fold in folds]))
        y_train.append(np.vstack([fold.y_train[None, :min_fold_size_train, ...] for fold in folds]))

        min_fold_size_test = np.min([fold.x_test.shape[0] for fold in folds])
        x_test.append(np.vstack([fold.x_test[None, :min_fold_size_test, ...] for fold in folds]))
        y_test.append(np.vstack([fold.y_test[None, :min_fold_size_test, ...] for fold in folds]))
    return TrainTestData(
        x_train=np.swapaxes(np.array(x_train), 0, 1),
        y_train=np.swapaxes(np.array(y_train), 0, 1),
        x_test=np.swapaxes(np.array(x_test), 0, 1),
        y_test=np.swapaxes(np.array(y_test), 0, 1),
    )


def _train(
    weights,
    x_train,
    y_train,
    number_of_epochs,
    kernel_class,
    kernel_parameters,
    kernel_parameter_name,
):
    return vmap(
        lambda weights_for_a_parameter, kernel_parameter: vmap(
            lambda w, x, y: Perceptron.train(
                w,
                kernel_class(x, **{kernel_parameter_name: kernel_parameter}),
                y,
                number_of_epochs,
            )
        )(weights_for_a_parameter, x_train, y_train)
    )(weights, kernel_parameters)


def _predict(
    weights, x_train, x, kernel_class, kernel_parameters, kernel_parameter_name
):
    return vmap(
        lambda weights_for_a_parameter, kernel_parameter: vmap(
            lambda w, x_train, x_i: Perceptron.predict(
                w,
                kernel_class(x_train, x_i, **{kernel_parameter_name: kernel_parameter}),
            )
        )(weights_for_a_parameter, x_train, x)
    )(weights, kernel_parameters)


def _performance(actuals, predictions) -> Performance:
    comparisons = vmap(
        lambda prediction: (
            jnp.argmax(prediction, axis=1) != jnp.argmax(actuals, axis=-1)
        )
    )(predictions)
    return Performance(
        mean=np.mean(np.mean(comparisons, axis=-1), axis=-1),
        stdev=np.std(np.mean(comparisons, axis=-1), axis=-1),
    )


def _performance_k_folds(actuals, predictions) -> np.ndarray:
    comparisons = vmap(
        lambda prediction: (
            jnp.argmax(prediction, axis=1) != jnp.argmax(actuals, axis=-1)
        )
    )(predictions)
    return np.mean(comparisons, axis=-1)


def _train_optimal_parameters(
        weights, kernel_parameters, x_train, y_train, kernel_class, kernel_parameter_name, number_of_epochs
):
    return vmap(
        lambda w, kernel_parameter, x, y: Perceptron.train(
                w,
                kernel_class(x, **{kernel_parameter_name: kernel_parameter}),
                y,
                number_of_epochs,
            )
    )(weights, kernel_parameters, x_train, y_train)


def _predict_optimal_parameters(
        weights, kernel_parameters, x_train, x_test, kernel_class, kernel_parameter_name
):
    predictions= vmap(
        lambda w, kernel_parameter, x, y: Perceptron.predict(
                w,
                kernel_class(x, y, **{kernel_parameter_name: kernel_parameter}),
            )
    )(weights, kernel_parameters, x_train, x_test)
    return np.swapaxes(predictions, 1, 2)


def _performance_optimal_parameters(actuals, predictions) -> Performance:
    comparisons = jnp.argmax(predictions, axis=-1) != jnp.argmax(actuals, axis=-1)
    return Performance(
        mean=np.mean(np.mean(comparisons, axis=-1)).reshape(1, 1),
        stdev=np.std(np.mean(comparisons, axis=-1)).reshape(1, 1),
    )


def q1(
    data: TrainTestData,
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    number_of_runs,
    number_of_epochs,
    df_performance_path,
):
    number_of_parameters = len(kernel_parameters)
    number_training_points = data.x_train.shape[1]
    number_classes = data.y_train.shape[2]

    weights = _train(
        weights=np.zeros(
            (number_of_parameters, number_of_runs, number_training_points, number_classes)
        ),
        x_train=data.x_train,
        y_train=data.y_train,
        number_of_epochs=number_of_epochs,
        kernel_class=kernel_class,
        kernel_parameters=kernel_parameters,
        kernel_parameter_name=kernel_parameter_name,
    )

    test_predictions = _predict(
        weights=weights,
        x_train=data.x_train,
        x=data.x_test,
        kernel_class=kernel_class,
        kernel_parameters=kernel_parameters,
        kernel_parameter_name=kernel_parameter_name,
    )

    train_predictions = _predict(
        weights=weights,
        x_train=data.x_train,
        x=data.x_train,
        kernel_class=kernel_class,
        kernel_parameters=kernel_parameters,
        kernel_parameter_name=kernel_parameter_name,
    )

    train_performance = _performance(data.y_train, train_predictions)
    test_performance = _performance(data.y_test, test_predictions)

    pd.concat(
        [
            train_performance.build_df(
                kernel_parameters=['%.2f' % x for x in kernel_parameters],
                kernel_parameter_name=kernel_parameter_name,
                index="Train",
            ),
            test_performance.build_df(
                kernel_parameters=['%.2f' % x for x in kernel_parameters],
                kernel_parameter_name=kernel_parameter_name,
                index="Test",
            ),
        ]
    ).to_csv(df_performance_path)


def q2(
    data: TrainTestData,
    kernel_class: BaseKernel,
    kernel_parameters,
    kernel_parameter_name,
    number_of_runs,
    number_of_epochs,
    number_of_folds,
    labels,
    df_performance_path,
):
    number_of_parameters = len(kernel_parameters)
    number_classes = data.y_train.shape[2]

    data_k_fold = _build_folds_data(data.x_train, data.y_train, number_of_folds)
    k_fold_number_training_points = data_k_fold.x_train.shape[2]
    weights_k_fold = np.zeros((
        number_of_folds, number_of_parameters, number_of_runs, k_fold_number_training_points, number_classes
    ))

    weights_k_fold = vmap(
        lambda weight_k_fold, x_train, y_train: _train(
            weight_k_fold,
            x_train,
            y_train,
            number_of_epochs,
            kernel_class=kernel_class,
            kernel_parameters=kernel_parameters,
            kernel_parameter_name=kernel_parameter_name,
        )
    )(weights_k_fold, data_k_fold.x_train, data_k_fold.y_train)

    test_predictions = vmap(
        lambda weight_k_fold, x_train, x_test: _predict(
            weight_k_fold,
            x_train=x_train,
            x=x_test,
            kernel_class=kernel_class,
            kernel_parameters=kernel_parameters,
            kernel_parameter_name=kernel_parameter_name,
        )
    )(weights_k_fold, data_k_fold.x_train, data_k_fold.x_test)

    test_performance = vmap(
        lambda y, prediction : _performance_k_folds(y, prediction)
    )(data_k_fold.y_test, test_predictions)
    avg_test_performance_per_run = np.mean(test_performance, axis=0)
    best_parameter_per_run = kernel_parameters[np.argmin(avg_test_performance_per_run, axis=0)]

    # train optimal weights
    number_training_points = data.x_train.shape[1]
    weights = _train_optimal_parameters(
        weights=np.zeros((len(best_parameter_per_run), number_training_points, number_classes)),
        kernel_parameters=best_parameter_per_run,
        x_train=data.x_train,
        y_train=data.y_train,
        kernel_class=kernel_class,
        kernel_parameter_name=kernel_parameter_name,
        number_of_epochs=number_of_epochs,
    )
    test_predictions = _predict_optimal_parameters(
        weights=weights,
        kernel_parameters=best_parameter_per_run,
        x_train=data.x_train,
        x_test=data.x_test,
        kernel_class=kernel_class,
        kernel_parameter_name=kernel_parameter_name,
    )
    test_performance = _performance_optimal_parameters(
        actuals=data.y_test,
        predictions=test_predictions
    )
    test_performance.build_df(
        kernel_parameters=[f"{np.mean(best_parameter_per_run)}±{np.std(best_parameter_per_run)}"],
        kernel_parameter_name=f"Mean Optimal {kernel_parameter_name}",
        index="Test Error Rate",
    ).to_csv(df_performance_path)

    # confusion_matrix = np.zeros((number_of_runs, number_classes, number_classes))
    # for i in range(number_of_runs):
    #     np.add.at(
    #         confusion_matrix[i, ...],
    #         (np.argmax(data.y_test[i, ...], axis=-1), np.argmax(test_predictions[i, ...], axis=-1)),
    #         1
    #     )
    #     np.fill_diagonal(confusion_matrix[i, ...], 0)
    # df = pd.DataFrame(
    #     confusion_matrix,
    #     columns=labels,
    # )
    # df.index = labels
