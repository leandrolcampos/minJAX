"""Library of primitive operations."""

from collections.abc import Sequence

import numpy as np

from minjax import primitives
from minjax.typing import Array, ArrayLike

__all__ = [
    "neg",
    "add",
    "mul",
    "greater",
    "less",
    "sin",
    "cos",
    "broadcast",
    "transpose",
    "reduce_sum",
]


def neg(x: ArrayLike) -> Array:
    """Returns numerical negative value element-wise."""
    return primitives.neg_p.bind(x)


def add(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns `x + y` element-wise."""
    return primitives.add_p.bind(x, y)


def mul(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns `x * y` element-wise."""
    return primitives.mul_p.bind(x, y)


def greater(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns the truth value of `x > y` element-wise."""
    return primitives.greater_p.bind(x, y)


def less(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns the truth value of `x < y` element-wise."""
    return primitives.less_p.bind(x, y)


def sin(x: ArrayLike) -> Array:
    """Computes the sine of `x` element-wise."""
    return primitives.sin_p.bind(x)


def cos(x: ArrayLike) -> Array:
    """Computes the cosine of `x` element-wise."""
    return primitives.cos_p.bind(x)


def broadcast(
    x: ArrayLike, shape: Sequence[int], axes: None | Sequence[int] = None
) -> Array:
    """Broadcasts `x` to the given `shape`, expanding the original shape if necessary.

    Args:
        x: The array to broadcast.
        shape: The shape of the desired array.
        axes: Positions where the original shape is expanded, if necessary.

    Returns:
        The broadcasted array.
    """
    return primitives.broadcast_p.bind(x, shape=shape, axes=axes)


def transpose(x: ArrayLike, perm: None | Sequence[int] = None) -> Array:
    """Permutes the dimensions of `x` according to `perm`.

    Args:
        x: The array to transpose.
        perm: The permutation of the dimensions of `x`. If not provided, defaults to
            `range(x.ndim)[::-1]`, which reverses the order of the axes.

    Returns:
        The transposed array.
    """
    return primitives.transpose_p.bind(x, perm=perm)


def reduce_sum(x: ArrayLike, axis: None | int | Sequence[int]) -> Array:
    """Computes the sum of `x` along the given `axis`."""
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    elif isinstance(axis, int):
        axis = (axis,)
    return primitives.reduce_sum_p.bind(x, axis=axis)
