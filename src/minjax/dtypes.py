"""Array types."""

from typing import Any

import numpy as np

__all__ = [
    "float0",
    "get_dtype",
]


float0 = np.dtype([("float0", np.void, 0)])


python_scalar_dtypes: dict[type, np.dtype] = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float32"),
}


numpy_scalar_dtypes: dict[type, np.dtype] = {
    np.bool_: np.dtype("bool"),
    np.int32: np.dtype("int32"),
    np.int64: np.dtype("int64"),
    np.float32: np.dtype("float32"),
    np.float64: np.dtype("float64"),
}


minjax_dtypes = [
    np.dtype("bool"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("float32"),
    np.dtype("float64"),
    float0,
]


def get_dtype(x: Any) -> np.dtype:
    """Returns the dtype object of a value."""
    typ = type(x)

    if typ in python_scalar_dtypes:
        return python_scalar_dtypes[typ]

    if typ in numpy_scalar_dtypes:
        return numpy_scalar_dtypes[typ]

    if getattr(x, "dtype", None) in minjax_dtypes:
        return x.dtype

    try:
        dtype = np.result_type(x)
    except TypeError as err:
        error_message = f"Cannot determine the dtype of the value {x}."
        raise TypeError(error_message) from err

    if dtype not in minjax_dtypes:
        error_message = f"Value {x} does not have a valid minJAX dtype: `{dtype}`."
        raise TypeError(error_message)

    return dtype
