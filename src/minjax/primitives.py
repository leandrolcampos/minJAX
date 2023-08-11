"""Primitive operations."""

from collections.abc import Sequence

import numpy as np

from minjax import core
from minjax.typing import Array, ArrayLike

__all__ = [
    "neg_p",
    "add_p",
    "mul_p",
    "greater_p",
    "less_p",
    "sin_p",
    "cos_p",
    "broadcast_p",
    "transpose_p",
    "reduce_sum_p",
]


neg_p = core.Primitive("neg")
neg_p.impl_rule = lambda x: np.negative(x)

add_p = core.Primitive("add")
add_p.impl_rule = lambda x, y: np.add(x, y)

mul_p = core.Primitive("mul")
mul_p.impl_rule = lambda x, y: np.multiply(x, y)

greater_p = core.Primitive("greater")
greater_p.impl_rule = lambda x, y: np.greater(x, y)

less_p = core.Primitive("less")
less_p.impl_rule = lambda x, y: np.less(x, y)

sin_p = core.Primitive("sin")
sin_p.impl_rule = lambda x: np.sin(x)

cos_p = core.Primitive("cos")
cos_p.impl_rule = lambda x: np.cos(x)


def broadcast_impl(
    x: ArrayLike, *, shape: Sequence[int], axes: None | Sequence[int] = None
) -> Array:
    if axes is not None:
        for axis in sorted(axes):
            x = np.expand_dims(x, axis)
    return np.broadcast_to(x, shape)


broadcast_p = core.Primitive("broadcast")
broadcast_p.impl_rule = broadcast_impl

transpose_p = core.Primitive("transpose")
transpose_p.impl_rule = lambda x, *, perm: np.transpose(x, axes=perm)

reduce_sum_p = core.Primitive("reduce_sum")
reduce_sum_p.impl_rule = lambda x, *, axis: np.sum(x, axis=axis)
