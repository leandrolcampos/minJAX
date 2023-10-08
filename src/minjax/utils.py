"""Utility functions for MinJax."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from minjax import core
from minjax.typing import Array, ArrayLike

__all__ = [
    "ones_like",
    "safe_zip",
    "unzip2",
    "zeros_like",
]


def zeros_like(
    x: ArrayLike, shape: None | Sequence[int] = None, dtype: None | np.dtype = None
) -> Array:
    aval = core.get_aval(x)
    shape = aval.shape if shape is None else shape
    dtype = aval.dtype if dtype is None else dtype
    return np.zeros(shape, dtype)


def ones_like(
    x: ArrayLike, shape: None | Sequence[int] = None, dtype: None | np.dtype = None
) -> Array:
    aval = core.get_aval(x)
    shape = aval.shape if shape is None else shape
    dtype = aval.dtype if dtype is None else dtype
    return np.ones(shape, dtype)


def unzip2(pairs: Sequence[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    lst1, lst2 = [], []
    for x, y in pairs:
        lst1.append(x)
        lst2.append(y)
    return lst1, lst2


def safe_zip(*args: Sequence[Any]) -> Sequence[tuple[Any, ...]]:
    fst, *rest = list(map(list, args))
    n = len(fst)
    for arg in rest:
        if len(arg) != n:
            error_message = "Inconsistent argument lengths."
            raise ValueError(error_message)
    return list(zip(*args, strict=True))
