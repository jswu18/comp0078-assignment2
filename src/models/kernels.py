from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import vmap


class BaseKernel(ABC):
    """
    Abstract Kernel class
    """

    @staticmethod
    @abstractmethod
    def _pre_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def post_gram(pre_gram: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def _kernel(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray, y: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Kernel evaluation for arbitrary number of x features and y features

        :param x: ndarray of shape (number_of_x_features, number_of_dimensions)
        :param y: ndarray of shape (number_of_y_features, number_of_dimensions)
        :param **kwargs: additonal parameters for the kernel
        :return: gram matrix k(x, y), if y is None then k(x,x) (number_of_x_features, number_of_y_features)
        """
        # compute k(x, x) if y is None
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        assert (
            x.shape[1] == y.shape[1]
        ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"

        gram = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            gram[i, :] = vmap(
                lambda y_i: self._kernel(x[i, :], y_i, **kwargs),
            )(y)
        return gram

    def pre_gram(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        # compute k(x, x) if y is None
        if y is None:
            y = x

        # add dimension when x is 1D, assume the vector is a single feature
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        assert (
            x.shape[1] == y.shape[1]
        ), f"Dimension Mismatch: {x.shape[1]=} != {y.shape[1]=}"
        pre_gram = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            pre_gram[i, :] = vmap(
                lambda y_i: self._pre_kernel(x[i, :], y_i),
            )(y)
        return pre_gram
        # return vmap(
        #         lambda x_i: vmap(
        #             lambda y_i: self._pre_kernel(x_i, y_i),
        #         )(y)
        #     )(x)


class GaussianKernel(BaseKernel):
    """
    The Gaussian Kernel defined as:
        k(x, y) = exp(-σ||x-y||_2^2)
    where σ>0.
    """

    @staticmethod
    def _l2_squared(x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the L2 norm, ||x-y||_2^2

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: the L2 norm of x-y
        """
        xx = jnp.dot(x.T, x)
        xy = jnp.dot(x.T, y)
        yy = jnp.dot(y.T, y)
        return (xx - 2 * xy + yy).reshape()

    @staticmethod
    def _pre_kernel(x: np.ndarray, y: np.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x - y, ord=2)

    @staticmethod
    def post_gram(pre_gram: np.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.exp(-kwargs["sigma"] * pre_gram**2)

    @staticmethod
    def _kernel(x: np.ndarray, y: np.ndarray, **kwargs) -> jnp.ndarray:
        """
        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: evaluation of Gaussian Kernel
        """
        return jnp.exp(-kwargs["sigma"] * jnp.linalg.norm(x - y, ord=2) ** 2)


class PolynomialKernel(BaseKernel):
    """
    The Polynomial Kernel defined as:
        k(x, y) = <x, y>^d
    where d is the degree of the polynomial
    """

    @staticmethod
    def _pre_kernel(x: np.ndarray, y: np.ndarray) -> jnp.ndarray:
        return jnp.dot(x.T, y)

    @staticmethod
    def post_gram(pre_gram: np.ndarray, **kwargs) -> jnp.ndarray:
        return jnp.power(pre_gram, kwargs["degree"])

    @staticmethod
    def _kernel(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :param degree: int indicating the degree of the polynomial
        :return: evaluation of Polynomial Kernel
        """
        return jnp.power(jnp.dot(x.T, y), kwargs["degree"]).item()
