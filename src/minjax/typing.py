"""Static type annotations."""

from minjax import core

__all__ = [
    "Array",
    "ArrayLike",
]


# Annotation for any minJAX array or tracer.
Array = core.Array

# Annotation for any value that is safe to implicitly cast to a minJAX array.
ArrayLike = core.ArrayLike
