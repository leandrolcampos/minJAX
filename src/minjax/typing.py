"""Static type annotations."""

from minjax import core

__all__ = [
    "Array",
    "ArrayLike",
]


# TODO: remove `np.ndarray` from `Array` after implementing `DeviceArray`.
# Annotation for any minJAX array or tracer.
Array = core.Array


# Annotation for any value that is safe to implicitly cast to a minJAX array.
ArrayLike = core.ArrayLike
