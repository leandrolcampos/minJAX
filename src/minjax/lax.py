"""Library of primitive operations."""

from collections.abc import Sequence

import numpy as np

from minjax import core
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
    return neg_p.bind1(x)


def add(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns `x + y` element-wise."""
    return add_p.bind1(x, y)


def mul(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns `x * y` element-wise."""
    return mul_p.bind1(x, y)


def greater(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns the truth value of `x > y` element-wise."""
    return greater_p.bind1(x, y)


def less(x: ArrayLike, y: ArrayLike) -> Array:
    """Returns the truth value of `x < y` element-wise."""
    return less_p.bind1(x, y)


def sin(x: ArrayLike) -> Array:
    """Computes the sine of `x` element-wise."""
    return sin_p.bind1(x)


def cos(x: ArrayLike) -> Array:
    """Computes the cosine of `x` element-wise."""
    return cos_p.bind1(x)


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
    return broadcast_p.bind1(x, shape=shape, axes=axes)


def transpose(x: ArrayLike, perm: None | Sequence[int] = None) -> Array:
    """Permutes the dimensions of `x` according to `perm`.

    Args:
        x: The array to transpose.
        perm: The permutation of the dimensions of `x`. If not provided, defaults to
            `range(x.ndim)[::-1]`, which reverses the order of the axes.

    Returns:
        The transposed array.
    """
    return transpose_p.bind1(x, perm=perm)


def reduce_sum(x: ArrayLike, axis: None | int | Sequence[int]) -> Array:
    """Computes the sum of `x` along the given `axis`."""
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    elif isinstance(axis, int):
        axis = (axis,)
    return reduce_sum_p.bind1(x, axis=axis)


# ------------------------------------- primitives -------------------------------------


neg_p = core.Primitive("neg")
neg_p.impl_rule = lambda x: [np.negative(x)]

add_p = core.Primitive("add")
add_p.impl_rule = lambda x, y: [np.add(x, y)]

mul_p = core.Primitive("mul")
mul_p.impl_rule = lambda x, y: [np.multiply(x, y)]

greater_p = core.Primitive("greater")
greater_p.impl_rule = lambda x, y: [np.greater(x, y)]

less_p = core.Primitive("less")
less_p.impl_rule = lambda x, y: [np.less(x, y)]

sin_p = core.Primitive("sin")
sin_p.impl_rule = lambda x: [np.sin(x)]

cos_p = core.Primitive("cos")
cos_p.impl_rule = lambda x: [np.cos(x)]


def broadcast_impl(
    x: ArrayLike, *, shape: Sequence[int], axes: None | Sequence[int] = None
) -> Sequence[Array]:
    if axes is not None:
        for axis in sorted(axes):
            x = np.expand_dims(x, axis)
    return [np.broadcast_to(x, shape)]


broadcast_p = core.Primitive("broadcast")
broadcast_p.impl_rule = broadcast_impl

transpose_p = core.Primitive("transpose")
transpose_p.impl_rule = lambda x, *, perm: [np.transpose(x, axes=perm)]

reduce_sum_p = core.Primitive("reduce_sum")
reduce_sum_p.impl_rule = lambda x, *, axis: [np.sum(x, axis=axis)]
