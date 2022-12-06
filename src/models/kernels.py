from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, tree_util, vmap


class BaseKernel(ABC):
    """
    Abstact Kernel class
    """

    @abstractmethod
    @jit
    def _kernel(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        Kernel implementation for single features

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: evaluation of kernel
        """

    @jit
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

        return vmap(
            lambda x_i: vmap(
                lambda y_i: self._kernel(x_i, y_i, **kwargs),
            )(y),
        )(x)

    @abstractmethod
    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        """
        To have JIT-compiled class methods by registering the type as a custom PyTree object.
        As referenced in:
        https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

        :return: A tuple containing dynamic and a dictionary containing static values
        """
        raise NotImplementedError("Needs to implement tree_flatten")

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Tuple) -> BaseKernel:
        """
        To have JIT-compiled class methods by registering the type as a custom PyTree object.
        As referenced in:
        https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

        :param aux_data: tuple containing dynamic values
        :param children: dictionary containing dynamic values
        :return: Class instance
        """
        raise NotImplementedError("Needs to implement tree_unflatten")


class GaussianKernel(BaseKernel):
    """
    The Gaussian Kernel defined as:
        k(x, y) = exp(-σ||x-y||_2^2)
    where σ>0.
    """

    @jit
    def _kernel(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: evaluation of Gaussian Kernel
        """
        return jnp.exp(-kwargs["sigma"] * jnp.power(jnp.linalg.norm(x - y, ord=2), 2))

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> GaussianKernel:
        return cls()


class PolynomialKernel(BaseKernel):
    """
    The Polynomial Kernel defined as:
        k(x, y) = <x, y>^d
    where d is the degree of the polynomial
    """

    @jit
    def _kernel(self, x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """
        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :param degree: int indicating the degree of the polynomial
        :return: evaluation of Polynomial Kernel
        """
        return jnp.power(jnp.dot(x.T, y), kwargs["degree"])

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> PolynomialKernel:
        return cls()


for KernelClass in [
    PolynomialKernel,
    GaussianKernel,
]:
    tree_util.register_pytree_node(
        KernelClass,
        KernelClass.tree_flatten,
        KernelClass.tree_unflatten,
    )
