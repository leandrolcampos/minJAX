"""Core machinery."""

import contextlib
import operator as op
import typing
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Generic, Protocol, TypeVar, Union

import numpy as np

from minjax import dtypes, lax

__all__ = [
    "ConcreteArray",
    "get_aval",
    "new_main",
    "Primitive",
    "ShapedArray",
    "Trace",
    "Tracer",
]

# mypy implicitly sets this variable to true when type checking.
MYPY = False

minjax_types: set[type] = {
    bool,
    int,
    float,
    np.bool_,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.ndarray,
}

Array = Union[np.ndarray, "Tracer"]

ArrayLike = (
    Array
    | bool
    | int
    | float
    | np.bool_
    | np.int32
    | np.int64
    | np.float32
    | np.float64
    | np.ndarray
)

TracerType = TypeVar("TracerType", bound=ArrayLike)


if not MYPY:

    class ImplRule(Protocol):
        def __call__(self, *args: ArrayLike, **params: Any) -> Sequence[Array]:
            """Impl rule that runs a primitive operation using some interpreter."""
            ...

else:
    ImplRule = Any


class Primitive:
    """Represents a primitive operation."""

    name: str
    impl_rule: None | ImplRule = None

    def __init__(self, name: str) -> None:
        self.name = name

    def bind1(self, *args: ArrayLike, **params: Any) -> Array:
        top_trace = find_top_trace(args)
        (out,) = self.bind_with_trace(top_trace, args, params)
        return out

    def bind_with_trace(
        self, trace: "Trace", args: Sequence[ArrayLike], params: dict[str, Any]
    ) -> Sequence[Array]:
        tracers = [trace.full_raise(x) for x in args]
        out = trace.process_primitive(self, tracers, params)
        return [full_lower(x) for x in out]


# -------------------------------------- tracing ---------------------------------------


class Trace(Generic[TracerType], metaclass=ABCMeta):
    main: "MainTrace"

    def __init__(self, main: "MainTrace") -> None:
        self.main = main

    @abstractmethod
    def pure(self, x: ArrayLike) -> TracerType:
        ...

    @abstractmethod
    def lift(self, x: ArrayLike) -> TracerType:
        ...

    def full_raise(self, x: ArrayLike) -> TracerType:
        if not isinstance(x, Tracer):
            if type(x) not in minjax_types:
                error_message = (
                    f"Value {x} does not have a valid minJAX type: `{type(x)}`."
                )
                raise TypeError(error_message)

            return self.pure(x)

        level = self.main.level
        if x._trace.main is self.main:
            # TODO: check if this cast is correct.
            return typing.cast(TracerType, x)

        if x._trace.main.level < level:
            return self.lift(x)

        if x._trace.main.level > level:
            error_message = f"Can't lift level {x._trace.main.level} to {level}."
            raise Exception(error_message)

        # (val._trace.main is not self.main) and (val._trace.level == level)
        error_message = f"Different traces at same level: {x._trace}, {self}."
        raise Exception(error_message)

    @abstractmethod
    def process_primitive(
        self,
        primitive: Primitive,
        tracers: Sequence[TracerType],
        params: dict[str, Any],
    ) -> Sequence[TracerType]:
        ...


class Tracer(metaclass=ABCMeta):
    _trace: Trace
    __array_priority__: int = 1000

    @property
    @abstractmethod
    def aval(self) -> "ShapedArray":
        ...

    def full_lower(self) -> Array:
        return self

    def __neg__(self) -> Array:
        return self.aval._neg(self)

    def __add__(self, other: ArrayLike) -> Array:
        return self.aval._add(self, other)

    def __radd__(self, other: ArrayLike) -> Array:  # type: ignore
        return self.aval._radd(self, other)

    def __mul__(self, other: ArrayLike) -> Array:
        return self.aval._mul(self, other)

    def __rmul__(self, other: ArrayLike) -> Array:  # type: ignore
        return self.aval._rmul(self, other)

    def __gt__(self, other: ArrayLike) -> Array:  # type: ignore
        return self.aval._gt(self, other)

    def __lt__(self, other: ArrayLike) -> Array:  # type: ignore
        return self.aval._lt(self, other)

    def __bool__(self) -> bool:
        return self.aval._bool(self)

    def __nonzero__(self) -> bool:
        return self.aval._nonzero(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.aval, name)
        except AttributeError:
            error_message = f"{self.__class__.__name__} has no attribute {name}."
            raise AttributeError(error_message) from None


def find_top_trace(xs: Sequence[ArrayLike]) -> Trace:
    top_main = max(
        (x._trace.main for x in xs if isinstance(x, Tracer)),
        default=trace_stack[0],
        key=op.attrgetter("level"),
    )

    if dynamic_trace is not None and dynamic_trace.level > top_main.level:
        top_main = dynamic_trace

    return top_main.trace_type(top_main)


def full_lower(x: Array) -> Array:
    if isinstance(x, Tracer):
        return x.full_lower()

    return x


class EvalTrace(Trace[ArrayLike]):
    def pure(self, x: ArrayLike) -> ArrayLike:
        return np.asarray(x, dtype=dtypes.get_dtype(x))

    lift = pure

    def process_primitive(
        self,
        primitive: Primitive,
        tracers: Sequence[ArrayLike],
        params: dict[str, Any],
    ) -> Sequence[ArrayLike]:
        if primitive.impl_rule is None:
            error_message = f"Primitive `{primitive.name}` has no implementation rule."
            raise NotImplementedError(error_message)

        return primitive.impl_rule(*tracers, **params)


class MainTrace:
    def __init__(
        self, level: int, trace_type: type[Trace], global_data: Any = None
    ) -> None:
        self.level = level
        self.trace_type = trace_type
        self.global_data = global_data


trace_stack: list[MainTrace] = [MainTrace(0, EvalTrace)]
dynamic_trace: None | MainTrace = None


@contextlib.contextmanager
def new_main(trace_type: type[Trace], global_data: Any = None) -> Iterator[MainTrace]:
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()


# ---------------------------------- abstract values -----------------------------------


class ShapedArray:
    array_abstraction_level: int = 1

    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @staticmethod
    def _neg(x: ArrayLike) -> Array:
        return lax.neg(x)

    @staticmethod
    def _add(x: ArrayLike, y: ArrayLike) -> Array:
        return lax.add(x, y)

    @staticmethod
    def _radd(x: ArrayLike, y: ArrayLike) -> Array:
        return lax.add(y, x)

    @staticmethod
    def _mul(x: ArrayLike, y: ArrayLike) -> Array:
        return lax.mul(x, y)

    @staticmethod
    def _rmul(x: ArrayLike, y: ArrayLike) -> Array:
        return lax.mul(y, x)

    @staticmethod
    def _gt(x: ArrayLike, y: ArrayLike) -> Array:
        return lax.greater(x, y)

    @staticmethod
    def _lt(x: ArrayLike, y: ArrayLike) -> Array:
        return lax.less(x, y)

    @staticmethod
    def _bool(x: Tracer) -> bool:
        return bool(x.aval)

    @staticmethod
    def _nonzero(x: Tracer) -> bool:
        return bool(x.aval)

    def str_short(self) -> str:
        return f"{self.dtype.name}[{','.join(str(d) for d in self.shape)}]"

    def __bool__(self) -> bool:
        error_message = "ShapedArray can't be unambiguously converted to bool."
        raise Exception(error_message)

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype))

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __repr__(self) -> str:
        return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"


class ConcreteArray(ShapedArray):
    array_abstraction_level = 2

    def __init__(self, val: np.ndarray) -> None:
        self.val = val
        self.shape = val.shape
        self.dtype = val.dtype

    def __bool__(self) -> bool:
        return bool(self.val)


def get_aval(x: ArrayLike) -> ShapedArray:
    if isinstance(x, Tracer):
        return x.aval

    if type(x) in minjax_types:
        return ConcreteArray(np.asarray(x, dtype=dtypes.get_dtype(x)))

    error_message = f"Value {x} does not have a valid minJAX type: `{type(x)}`."
    raise TypeError(error_message)
