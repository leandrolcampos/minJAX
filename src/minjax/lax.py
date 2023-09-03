"""Library of primitive operations."""

from collections.abc import Sequence

import numpy as np

from minjax import core, dtypes, utils
from minjax.interpreters import ad
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


def neg_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x,) = primals
    (x_dot,) = tangents
    return [neg(x)], [neg(x_dot)]


neg_p = core.Primitive("neg")
neg_p.impl_rule = lambda x: [np.negative(x)]
ad.register_differentiation(neg_p, neg_jvp)


def add_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x, y) = primals
    (x_dot, y_dot) = tangents
    return [x + y], [x_dot + y_dot]


add_p = core.Primitive("add")
add_p.impl_rule = lambda x, y: [np.add(x, y)]
ad.register_differentiation(add_p, add_jvp)


def mul_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x, y) = primals
    (x_dot, y_dot) = tangents
    return [x * y], [x_dot * y + x * y_dot]


mul_p = core.Primitive("mul")
mul_p.impl_rule = lambda x, y: [np.multiply(x, y)]
ad.register_differentiation(mul_p, mul_jvp)


def greater_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x, y) = primals
    del tangents
    primal_out = greater(x, y)
    return [primal_out], [utils.zeros_like(primal_out, dtype=dtypes.float0)]


greater_p = core.Primitive("greater")
greater_p.impl_rule = lambda x, y: [np.greater(x, y)]
ad.register_differentiation(greater_p, greater_jvp)


def less_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x, y) = primals
    del tangents
    primal_out = less(x, y)
    return [primal_out], [utils.zeros_like(primal_out, dtype=dtypes.float0)]


less_p = core.Primitive("less")
less_p.impl_rule = lambda x, y: [np.less(x, y)]
ad.register_differentiation(less_p, less_jvp)


def sin_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x,) = primals
    (x_dot,) = tangents
    return [sin(x)], [cos(x) * x_dot]


sin_p = core.Primitive("sin")
sin_p.impl_rule = lambda x: [np.sin(x)]
ad.register_differentiation(sin_p, sin_jvp)


def cos_jvp(
    primals: Sequence[Array], tangents: Sequence[Array]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x,) = primals
    (x_dot,) = tangents
    return [cos(x)], [-sin(x) * x_dot]


cos_p = core.Primitive("cos")
cos_p.impl_rule = lambda x: [np.cos(x)]
ad.register_differentiation(cos_p, cos_jvp)


def broadcast_jvp(
    primals: Sequence[Array],
    tangents: Sequence[Array],
    *,
    shape: Sequence[int],
    axes: None | Sequence[int] = None,
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x,) = primals
    (x_dot,) = tangents
    primal_out = broadcast(x, shape=shape, axes=axes)
    tangent_out = broadcast(x_dot, shape=shape, axes=axes)
    return [primal_out], [tangent_out]


def broadcast_impl(
    x: ArrayLike, *, shape: Sequence[int], axes: None | Sequence[int] = None
) -> Sequence[Array]:
    if axes is not None:
        for axis in sorted(axes):
            x = np.expand_dims(x, axis)
    return [np.broadcast_to(x, shape)]


broadcast_p = core.Primitive("broadcast")
broadcast_p.impl_rule = broadcast_impl
ad.register_differentiation(broadcast_p, broadcast_jvp)


def transpose_jvp(
    primals: Sequence[Array], tangents: Sequence[Array], *, perm: Sequence[int]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x,) = primals
    (x_dot,) = tangents
    return [transpose(x, perm=perm)], [transpose(x_dot, perm=perm)]


transpose_p = core.Primitive("transpose")
transpose_p.impl_rule = lambda x, *, perm: [np.transpose(x, axes=perm)]
ad.register_differentiation(transpose_p, transpose_jvp)


def reduce_sum_jvp(
    primals: Sequence[Array], tangents: Sequence[Array], *, axis: int | Sequence[int]
) -> tuple[Sequence[Array], Sequence[Array]]:
    (x,) = primals
    (x_dot,) = tangents
    return [reduce_sum(x, axis=axis)], [reduce_sum(x_dot, axis=axis)]


reduce_sum_p = core.Primitive("reduce_sum")
reduce_sum_p.impl_rule = lambda x, *, axis: [np.sum(x, axis=axis)]
ad.register_differentiation(reduce_sum_p, reduce_sum_jvp)
